from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


_EPS = np.float32(1e-12)


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Vectorized safe divide -> float32, NaN where denominator is zero/non-finite."""
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    out = np.full(num.shape, np.nan, dtype=np.float32)
    mask = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-12)
    out[mask] = (num[mask] / den[mask]).astype(np.float32)
    return out


def _required_columns(levels: int) -> list[str]:
    cols = [
        "RIC",
        "bucketEnd",
        "mid",
        "numSharesTradedPrim",
        "numSharesTradedPrimBuy",
        "numSharesTradedPrimSell",
        "numSharesTradedPrimUnknown",
        "numSharesTradedAllLit",
        "numSharesTradedAllLitBuy",
        "numSharesTradedAllLitSell",
        "numSharesTradedAllLitUnknown",
    ]
    for i in range(levels):
        cols += [
            f"bidPx_{i}", f"bidQty_{i}", f"bidOrderCount_{i}",
            f"offerPx_{i}", f"offerQty_{i}", f"offerOrderCount_{i}",
        ]
    return cols


def _ensure_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns ({len(missing)}): {missing}")


def _to_f64(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """Fast numeric extraction. Assumes input columns are already numeric."""
    return df.loc[:, cols].to_numpy(dtype=np.float64, copy=False)


def _compute_features_one_chunk(
    chunk: pd.DataFrame,
    *,
    levels: int,
    group_cols: tuple[str, ...],
    time_col: str,
    depth_ks: tuple[int, ...],
    rolling_windows: tuple[int, ...],
    volatility_minutes: tuple[int, ...],
    rows_per_minute: int,
    target_horizon_minutes: int,
    add_target_mid_return: bool,
    add_target_log_return: bool,
    add_target_log_return_volnorm: bool,
    distance_decay: float,
) -> pd.DataFrame:
    """Compute features for a chunk that contains complete RICs (never split one RIC across workers)."""
    work = chunk.copy(deep=False)
    sort_cols = list(group_cols) + [time_col]
    work = work.sort_values(sort_cols, kind="mergesort")

    n = len(work)
    out = pd.DataFrame(index=work.index)
    out["__row_pos__"] = work["__row_pos__"].to_numpy(copy=False)

    bid_px_cols = [f"bidPx_{i}" for i in range(levels)]
    ask_px_cols = [f"offerPx_{i}" for i in range(levels)]
    bid_q_cols = [f"bidQty_{i}" for i in range(levels)]
    ask_q_cols = [f"offerQty_{i}" for i in range(levels)]
    bid_n_cols = [f"bidOrderCount_{i}" for i in range(levels)]
    ask_n_cols = [f"offerOrderCount_{i}" for i in range(levels)]

    bp = _to_f64(work, bid_px_cols)
    ap = _to_f64(work, ask_px_cols)
    bq = _to_f64(work, bid_q_cols)
    aq = _to_f64(work, ask_q_cols)
    bn = _to_f64(work, bid_n_cols)
    an = _to_f64(work, ask_n_cols)

    mid = pd.to_numeric(work["mid"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    best_bid, best_ask = bp[:, 0], ap[:, 0]
    spread = best_ask - best_bid

    group_key = pd.MultiIndex.from_frame(work.loc[:, list(group_cols)])
    gid, _ = pd.factorize(group_key, sort=False)
    same_prev = np.r_[False, gid[1:] == gid[:-1]]

    def prev2d(x: np.ndarray) -> np.ndarray:
        z = np.empty_like(x, dtype=np.float64)
        z[0, :] = np.nan
        z[1:, :] = x[:-1, :]
        z[~same_prev, :] = np.nan
        return z

    def prev1d(x: np.ndarray) -> np.ndarray:
        z = np.empty_like(x, dtype=np.float64)
        z[0] = np.nan
        z[1:] = x[:-1]
        z[~same_prev] = np.nan
        return z

    def shift1d(x: np.ndarray, periods: int) -> np.ndarray:
        """Group-aware shift without pandas groupby; positive=lag, negative=lead."""
        x = np.asarray(x, dtype=np.float64)
        z = np.full(x.shape, np.nan, dtype=np.float64)
        if periods == 0:
            z[:] = x
            return z
        k = abs(int(periods))
        if k >= len(x):
            return z
        if periods > 0:
            same_group = gid[k:] == gid[:-k]
            dst = z[k:]
            dst[same_group] = x[:-k][same_group]
        else:
            same_group = gid[:-k] == gid[k:]
            dst = z[:-k]
            dst[same_group] = x[k:][same_group]
        return z

    def grouped_realized_vol(log_returns: np.ndarray, window_rows: int) -> np.ndarray:
        """Trailing unannualized realized vol = sqrt(sum(log_return^2))."""
        if window_rows < 1:
            raise ValueError("window_rows must be >= 1")
        result = np.full(len(log_returns), np.nan, dtype=np.float32)
        starts = np.r_[0, np.flatnonzero(gid[1:] != gid[:-1]) + 1]
        ends = np.r_[starts[1:], len(gid)]
        for start, end in zip(starts, ends):
            m = int(end - start)
            if m < window_rows:
                continue
            v = np.asarray(log_returns[start:end], dtype=np.float64)
            finite = np.isfinite(v)
            sq = np.where(finite, v * v, 0.0)
            csum = np.cumsum(sq, dtype=np.float64)
            ccount = np.cumsum(finite.astype(np.int32), dtype=np.int32)
            win_sum = csum[window_rows - 1:].copy()
            win_count = ccount[window_rows - 1:].copy()
            if window_rows < m:
                win_sum[1:] -= csum[:-window_rows]
                win_count[1:] -= ccount[:-window_rows]
            vals = np.full(len(win_sum), np.nan, dtype=np.float32)
            good = win_count == window_rows
            vals[good] = np.sqrt(win_sum[good]).astype(np.float32)
            result[start + window_rows - 1:end] = vals
        return result

    is_locked = np.isfinite(best_bid) & np.isfinite(best_ask) & (best_bid == best_ask)
    is_crossed = np.isfinite(best_bid) & np.isfinite(best_ask) & (best_bid > best_ask)
    valid_book = (
        np.isfinite(best_bid) & np.isfinite(best_ask) &
        (best_bid > 0) & (best_ask > 0) &
        (best_bid < best_ask)
    )
    if "bookIsCrossed" in work.columns:
        vendor_cross = work["bookIsCrossed"].fillna(False).astype(bool).to_numpy()
        valid_book &= ~vendor_cross

    out["lob_is_locked"] = is_locked.astype(np.int8)
    out["lob_is_strict_crossed"] = is_crossed.astype(np.int8)
    out["lob_valid_book"] = valid_book.astype(np.int8)
    out["lob_spread_bps"] = (_safe_div(spread, mid) * np.float32(1e4)).astype(np.float32)

    bid_cum = np.cumsum(bq, axis=1)
    ask_cum = np.cumsum(aq, axis=1)
    bid_n_cum = np.cumsum(bn, axis=1)
    ask_n_cum = np.cumsum(an, axis=1)

    for k in depth_ks:
        j = k - 1
        bd, ad = bid_cum[:, j], ask_cum[:, j]
        bnc, anc = bid_n_cum[:, j], ask_n_cum[:, j]
        out[f"lob_bid_depth_{k}"] = bd.astype(np.float32)
        out[f"lob_ask_depth_{k}"] = ad.astype(np.float32)
        out[f"lob_total_depth_{k}"] = (bd + ad).astype(np.float32)
        out[f"lob_qi_{k}"] = _safe_div(bd - ad, bd + ad)
        out[f"lob_oci_{k}"] = _safe_div(bnc - anc, bnc + anc)

    weights = np.exp(-distance_decay * np.arange(levels, dtype=np.float64))
    wb = bq @ weights
    wa = aq @ weights
    out[f"lob_wqi_{levels}"] = _safe_div(wb - wa, wb + wa)
    out["lob_avg_bid_order_size_1"] = _safe_div(bq[:, 0], bn[:, 0])
    out["lob_avg_ask_order_size_1"] = _safe_div(aq[:, 0], an[:, 0])
    out[f"lob_avg_bid_order_size_{levels}"] = _safe_div(bid_cum[:, -1], bid_n_cum[:, -1])
    out[f"lob_avg_ask_order_size_{levels}"] = _safe_div(ask_cum[:, -1], ask_n_cum[:, -1])

    micro = _safe_div(best_ask * bq[:, 0] + best_bid * aq[:, 0], bq[:, 0] + aq[:, 0]).astype(np.float32)
    out["lob_microprice"] = micro
    out["lob_microprice_dev_bps"] = (_safe_div(micro.astype(np.float64) - mid, mid) * np.float32(1e4)).astype(np.float32)

    out[f"lob_bid_l1_share_{levels}"] = _safe_div(bq[:, 0], bid_cum[:, -1])
    out[f"lob_ask_l1_share_{levels}"] = _safe_div(aq[:, 0], ask_cum[:, -1])
    near = min(2, levels)
    out[f"lob_bid_near_share_{near}_{levels}"] = _safe_div(bid_cum[:, near - 1], bid_cum[:, -1])
    out[f"lob_ask_near_share_{near}_{levels}"] = _safe_div(ask_cum[:, near - 1], ask_cum[:, -1])
    bid_share = np.divide(bq, bid_cum[:, [-1]], out=np.zeros_like(bq), where=bid_cum[:, [-1]] != 0)
    ask_share = np.divide(aq, ask_cum[:, [-1]], out=np.zeros_like(aq), where=ask_cum[:, [-1]] != 0)
    out[f"lob_bid_hhi_{levels}"] = np.sum(bid_share * bid_share, axis=1).astype(np.float32)
    out[f"lob_ask_hhi_{levels}"] = np.sum(ask_share * ask_share, axis=1).astype(np.float32)

    if levels > 1:
        bid_gaps = bp[:, :-1] - bp[:, 1:]
        ask_gaps = ap[:, 1:] - ap[:, :-1]
        max_bid_gap = np.nanmax(bid_gaps, axis=1)
        max_ask_gap = np.nanmax(ask_gaps, axis=1)
        out[f"lob_max_bid_gap_bps_{levels}"] = (_safe_div(max_bid_gap, mid) * np.float32(1e4)).astype(np.float32)
        out[f"lob_max_ask_gap_bps_{levels}"] = (_safe_div(max_ask_gap, mid) * np.float32(1e4)).astype(np.float32)

    wbid_px = _safe_div(np.sum(bp * bq, axis=1), np.sum(bq, axis=1)).astype(np.float64)
    wask_px = _safe_div(np.sum(ap * aq, axis=1), np.sum(aq, axis=1)).astype(np.float64)
    book_center = 0.5 * (wbid_px + wask_px)
    out[f"lob_book_center_dev_bps_{levels}"] = (_safe_div(book_center - mid, mid) * np.float32(1e4)).astype(np.float32)

    prim = pd.to_numeric(work["numSharesTradedPrim"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    pbuy = pd.to_numeric(work["numSharesTradedPrimBuy"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    psell = pd.to_numeric(work["numSharesTradedPrimSell"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    punk = pd.to_numeric(work["numSharesTradedPrimUnknown"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    allv = pd.to_numeric(work["numSharesTradedAllLit"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    abuy = pd.to_numeric(work["numSharesTradedAllLitBuy"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    asell = pd.to_numeric(work["numSharesTradedAllLitSell"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    aunk = pd.to_numeric(work["numSharesTradedAllLitUnknown"], errors="coerce").to_numpy(dtype=np.float64, copy=False)

    ti_prim = _safe_div(pbuy - psell, pbuy + psell)
    ti_all = _safe_div(abuy - asell, abuy + asell)
    out["flow_ti_prim"] = ti_prim
    out["flow_ti_alllit"] = ti_all
    out["flow_ti_prim_minus_alllit"] = (ti_prim - ti_all).astype(np.float32)
    out["flow_signed_vol_prim"] = (pbuy - psell).astype(np.float32)
    out["flow_signed_vol_alllit"] = (abuy - asell).astype(np.float32)
    out["flow_primary_volume_share"] = _safe_div(prim, allv)
    out["flow_unknown_ratio_prim"] = _safe_div(punk, prim)
    out["flow_unknown_ratio_alllit"] = _safe_div(aunk, allv)

    bp0, ap0, bq0, aq0 = prev2d(bp), prev2d(ap), prev2d(bq), prev2d(aq)
    bid_contrib = ((bp >= bp0) * bq) - ((bp <= bp0) * bq0)
    ask_contrib = -((ap <= ap0) * aq) + ((ap >= ap0) * aq0)
    ofi = bid_contrib + ask_contrib
    prev_valid = prev1d(valid_book.astype(np.float64)) == 1.0
    valid_transition = same_prev & valid_book & prev_valid
    ofi[~valid_transition, :] = np.nan

    for level in range(levels):
        out[f"ofi_level_{level}"] = ofi[:, level].astype(np.float32)
    out["ofi_l1"] = ofi[:, 0].astype(np.float32)
    out[f"mlofi_sum_{levels}"] = np.nansum(ofi, axis=1).astype(np.float32)
    out[f"mlofi_weighted_{levels}"] = np.nansum(ofi * weights[None, :], axis=1).astype(np.float32)

    total_depth = bid_cum[:, -1] + ask_cum[:, -1]
    out[f"mlofi_norm_depth_{levels}"] = _safe_div(out[f"mlofi_weighted_{levels}"].to_numpy(dtype=np.float64), total_depth)

    prev_bq = prev2d(bq)
    prev_aq = prev2d(aq)
    same_bid_px = np.isfinite(bp0) & np.isfinite(bp) & (bp == bp0)
    same_ask_px = np.isfinite(ap0) & np.isfinite(ap) & (ap == ap0)
    bid_refill = np.where(same_bid_px, np.maximum(bq - prev_bq, 0.0), np.nan)
    ask_refill = np.where(same_ask_px, np.maximum(aq - prev_aq, 0.0), np.nan)
    bid_cancel = np.where(same_bid_px, np.maximum(prev_bq - bq, 0.0), np.nan)
    ask_cancel = np.where(same_ask_px, np.maximum(prev_aq - aq, 0.0), np.nan)
    for arr in (bid_refill, ask_refill, bid_cancel, ask_cancel):
        arr[~valid_transition, :] = np.nan
    out[f"lob_bid_refill_{levels}"] = np.nansum(bid_refill, axis=1).astype(np.float32)
    out[f"lob_ask_refill_{levels}"] = np.nansum(ask_refill, axis=1).astype(np.float32)
    out[f"lob_bid_cancel_{levels}"] = np.nansum(bid_cancel, axis=1).astype(np.float32)
    out[f"lob_ask_cancel_{levels}"] = np.nansum(ask_cancel, axis=1).astype(np.float32)
    out[f"lob_refill_imbalance_{levels}"] = _safe_div(
        out[f"lob_bid_refill_{levels}"].to_numpy(dtype=np.float64) - out[f"lob_ask_refill_{levels}"].to_numpy(dtype=np.float64),
        out[f"lob_bid_refill_{levels}"].to_numpy(dtype=np.float64) + out[f"lob_ask_refill_{levels}"].to_numpy(dtype=np.float64),
    )
    out[f"lob_cancel_imbalance_{levels}"] = _safe_div(
        out[f"lob_bid_cancel_{levels}"].to_numpy(dtype=np.float64) - out[f"lob_ask_cancel_{levels}"].to_numpy(dtype=np.float64),
        out[f"lob_bid_cancel_{levels}"].to_numpy(dtype=np.float64) + out[f"lob_ask_cancel_{levels}"].to_numpy(dtype=np.float64),
    )

    mid_prev = prev1d(mid)
    log_ret_1 = np.full(n, np.nan, dtype=np.float64)
    valid_log_ret = (
        valid_transition & np.isfinite(mid) & np.isfinite(mid_prev) &
        (mid > 0) & (mid_prev > 0)
    )
    log_ret_1[valid_log_ret] = np.log(mid[valid_log_ret] / mid_prev[valid_log_ret])

    vol_minutes_needed = set(volatility_minutes)
    if add_target_log_return_volnorm:
        vol_minutes_needed.add(target_horizon_minutes)

    past_vol_cache: dict[int, np.ndarray] = {}
    for minutes in sorted(vol_minutes_needed):
        window_rows = int(minutes * rows_per_minute)
        vol = grouped_realized_vol(log_ret_1, window_rows)
        past_vol_cache[minutes] = vol
        if minutes in volatility_minutes:
            out[f"past_vol_{minutes}m"] = vol

    if add_target_mid_return or add_target_log_return or add_target_log_return_volnorm:
        target_rows = int(target_horizon_minutes * rows_per_minute)
        future_mid = shift1d(mid, -target_rows)
        future_valid = shift1d(valid_book.astype(np.float64), -target_rows) == 1.0
        target_mask = (
            valid_book & future_valid &
            np.isfinite(mid) & np.isfinite(future_mid) &
            (mid > 0) & (future_mid > 0)
        )

        if add_target_mid_return:
            simple_ret = np.full(n, np.nan, dtype=np.float64)
            simple_ret[target_mask] = future_mid[target_mask] / mid[target_mask] - 1.0
            out[f"mid_return_t_plus_{target_horizon_minutes}"] = simple_ret.astype(np.float32)

        target_log_ret = np.full(n, np.nan, dtype=np.float64)
        target_log_ret[target_mask] = np.log(future_mid[target_mask] / mid[target_mask])

        if add_target_log_return:
            out[f"target_log_return_{target_horizon_minutes}m"] = target_log_ret.astype(np.float32)

        if add_target_log_return_volnorm:
            out[f"target_log_return_{target_horizon_minutes}m_volnorm"] = _safe_div(
                target_log_ret, past_vol_cache[target_horizon_minutes]
            )

    tmp = pd.DataFrame(
        {
            "gid": gid,
            "qi1": out["lob_qi_1"].to_numpy(dtype=np.float32, copy=False),
            "ti_all": ti_all,
            "mlofi_w": out[f"mlofi_weighted_{levels}"].to_numpy(dtype=np.float32, copy=False),
            "ret1": _safe_div(mid - mid_prev, mid_prev),
        },
        index=work.index,
    )
    tmp["ret1_sq"] = (tmp["ret1"].astype(np.float64) ** 2).astype(np.float32)
    g = tmp.groupby("gid", sort=False, observed=True)
    mid_grouped = work.groupby(list(group_cols), sort=False, observed=True)["mid"]

    for w in rolling_windows:
        out[f"roll_qi1_mean_{w}"] = g["qi1"].rolling(w, min_periods=w).mean().reset_index(level=0, drop=True).astype(np.float32)
        out[f"roll_ti_alllit_mean_{w}"] = g["ti_all"].rolling(w, min_periods=w).mean().reset_index(level=0, drop=True).astype(np.float32)
        out[f"roll_mlofi_w_sum_{w}"] = g["mlofi_w"].rolling(w, min_periods=w).sum().reset_index(level=0, drop=True).astype(np.float32)
        rv2 = g["ret1_sq"].rolling(w, min_periods=w).sum().reset_index(level=0, drop=True).to_numpy(dtype=np.float64, copy=False)
        out[f"roll_realized_vol_{w}"] = np.sqrt(rv2).astype(np.float32)
        mid_w_prev = mid_grouped.shift(w).to_numpy(dtype=np.float64, copy=False)
        ret_w = _safe_div(mid - mid_w_prev, mid_w_prev)
        ret_w[~valid_book] = np.nan
        out[f"dyn_mid_ret_{w}"] = ret_w

    return out.reset_index(drop=True)


def _partition_by_ric(df: pd.DataFrame, n_parts: int) -> list[pd.DataFrame]:
    """Partition by complete RIC sets so no time series is split across processes."""
    rics = pd.unique(df["RIC"])
    n_parts = max(1, min(int(n_parts), len(rics)))
    ric_parts = np.array_split(rics, n_parts)
    parts: list[pd.DataFrame] = []
    ric_series = df["RIC"]
    for rp in ric_parts:
        mask = ric_series.isin(rp)
        parts.append(df.loc[mask])
    return parts


def add_lob_features(
    modeling_df: pd.DataFrame,
    *,
    levels: int = 8,
    group_cols: Sequence[str] = ("RIC", "date_dt"),
    time_col: str = "bucketEnd",
    depth_ks: Sequence[int] = (1, 3, 5, 8),
    rolling_windows: Sequence[int] = (3, 10, 30),
    volatility_minutes: Sequence[int] = (5, 15, 30, 60),
    rows_per_minute: int = 1,
    target_horizon_minutes: int = 30,
    add_target_mid_return: bool = False,
    add_target_log_return: bool = False,
    add_target_log_return_volnorm: bool = False,
    add_target_log_return_30m: Optional[bool] = None,
    add_target_log_return_30m_volnorm: Optional[bool] = None,
    distance_decay: float = 0.5,
    n_jobs: int = -1,
    chunks_per_worker: int = 1,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Add core LOB/order-flow features and optional configurable-horizon targets."""
    if not isinstance(modeling_df, pd.DataFrame):
        raise TypeError("modeling_df must be a pandas DataFrame")
    if len(modeling_df) == 0:
        return modeling_df

    group_cols = tuple(group_cols)
    depth_ks = tuple(sorted(set(int(k) for k in depth_ks)))
    rolling_windows = tuple(sorted(set(int(w) for w in rolling_windows)))
    volatility_minutes = tuple(sorted(set(int(m) for m in volatility_minutes)))
    rows_per_minute = int(rows_per_minute)
    target_horizon_minutes = int(target_horizon_minutes)

    if rows_per_minute < 1:
        raise ValueError("rows_per_minute must be >= 1")
    if target_horizon_minutes < 1:
        raise ValueError("target_horizon_minutes must be >= 1")
    if any(m < 1 for m in volatility_minutes):
        raise ValueError("volatility_minutes must contain positive integers")
    if any(k < 1 or k > levels for k in depth_ks):
        raise ValueError(f"depth_ks must be between 1 and levels={levels}; got {depth_ks}")
    if any(w < 2 for w in rolling_windows):
        raise ValueError("rolling_windows must be >= 2")

    if add_target_log_return_30m is not None:
        if target_horizon_minutes != 30:
            raise ValueError("Legacy add_target_log_return_30m is only valid when target_horizon_minutes=30")
        add_target_log_return = bool(add_target_log_return_30m)
    if add_target_log_return_30m_volnorm is not None:
        if target_horizon_minutes != 30:
            raise ValueError("Legacy add_target_log_return_30m_volnorm is only valid when target_horizon_minutes=30")
        add_target_log_return_volnorm = bool(add_target_log_return_30m_volnorm)

    required = _required_columns(levels)
    required += [c for c in group_cols if c not in required]
    if time_col not in required:
        required.append(time_col)
    _ensure_columns(modeling_df, required)

    work = modeling_df.copy(deep=False)
    work = work.assign(__row_pos__=np.arange(len(work), dtype=np.int64))

    cpu = os.cpu_count() or 1
    if n_jobs == -1:
        n_jobs = max(1, cpu - 1)
    n_jobs = max(1, min(int(n_jobs), cpu))

    kwargs = dict(
        levels=levels,
        group_cols=group_cols,
        time_col=time_col,
        depth_ks=depth_ks,
        rolling_windows=rolling_windows,
        volatility_minutes=volatility_minutes,
        rows_per_minute=rows_per_minute,
        target_horizon_minutes=target_horizon_minutes,
        add_target_mid_return=bool(add_target_mid_return),
        add_target_log_return=bool(add_target_log_return),
        add_target_log_return_volnorm=bool(add_target_log_return_volnorm),
        distance_decay=float(distance_decay),
    )

    if n_jobs == 1 or work["RIC"].nunique(dropna=False) <= 1:
        features = _compute_features_one_chunk(work, **kwargs)
    else:
        n_parts = min(work["RIC"].nunique(dropna=False), n_jobs * max(1, int(chunks_per_worker)))
        parts = _partition_by_ric(work, n_parts)
        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(_compute_features_one_chunk, p, **kwargs) for p in parts]
            iterator = as_completed(futures)
            if show_progress and tqdm is not None:
                iterator = tqdm(iterator, total=len(futures), desc="LOB feature chunks", unit="chunk")
            for fut in iterator:
                results.append(fut.result())
        features = pd.concat(results, axis=0, ignore_index=True, copy=False)

    features.sort_values("__row_pos__", inplace=True, kind="mergesort")
    feature_cols = [c for c in features.columns if c != "__row_pos__"]
    feat_values = features.loc[:, feature_cols].reset_index(drop=True)
    for c in feature_cols:
        modeling_df[c] = feat_values[c].to_numpy(copy=False)
    return modeling_df
