# LOB balanced IC checkpoint selection v022

This directory preserves the exact verified LOB training bundle generated on 2026-08-22.

## Key behavior

- A deployable checkpoint must have both TRAIN IC and TEST IC above their configured minima.
- Eligible checkpoints are ranked by:
  `balanced_ic = min(train_ic, test_ic)^2 / max(train_ic, test_ic)`.
- Repeated seeds are ranked as individual runs; seed clustering/stability is not part of primary model ranking.
- The status notebook includes TRAIN IC, TEST IC and Balanced IC by epoch, with learning rate on a secondary y-axis.
- The v021 generic `TRAINING_HPARAM_GRID` behavior is retained.

## Exact verified bundle

The exact ZIP is stored losslessly as Base64 chunks under `verified_bundle/` because the connected GitHub write interface accepts UTF-8 content rather than local binary-file attachments.

Reconstruct and verify it with:

```bash
python reconstruct_verified_zip.py
```

Expected ZIP SHA256:

`687e6eb5b8424a5e180b1d2e31eedb0755ed5927a92ebbeb3cf48f7b64d69e67`

The reconstructed file is:

`lob_balanced_ic_checkpoint_selection_v1_0_cpp_v022_verified_20260822.zip`

See `PACKAGE_MANIFEST_lob_balanced_ic_checkpoint_selection_v1_0_cpp_v022.txt`, `CODE_DIFF_SUMMARY_v022.txt`, and `file_hashes.txt` for contents and verification details.
