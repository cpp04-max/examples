#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
from pathlib import Path

EXPECTED_SHA256 = "687e6eb5b8424a5e180b1d2e31eedb0755ed5927a92ebbeb3cf48f7b64d69e67"
OUTPUT_NAME = "lob_balanced_ic_checkpoint_selection_v1_0_cpp_v022_verified_20260822.zip"

root = Path(__file__).resolve().parent
parts_dir = root / "verified_bundle"
parts = sorted(parts_dir.glob("part*.b64"))
if not parts:
    raise SystemExit("No verified_bundle/part*.b64 files found")

encoded = b"".join(p.read_bytes() for p in parts)
payload = base64.b64decode(encoded, validate=True)
actual = hashlib.sha256(payload).hexdigest()
if actual != EXPECTED_SHA256:
    raise SystemExit(
        f"SHA256 mismatch: expected {EXPECTED_SHA256}, got {actual}"
    )

out = root / OUTPUT_NAME
out.write_bytes(payload)
print(f"Wrote {out}")
print(f"SHA256 {actual}")
