"""V7-C03: pretty-print self-describing checkpoint metadata.

Usage: python -m cppmega_v4.tools.ckpt_inspect FILE
"""

from __future__ import annotations

import argparse
import json
import sys

from cppmega_v4.runner.stages import read_ckpt_metadata


def main() -> int:
    p = argparse.ArgumentParser(prog="cppmega_v4.tools.ckpt_inspect")
    p.add_argument("file", help="path to safetensors checkpoint")
    args = p.parse_args()

    meta = read_ckpt_metadata(args.file)
    if meta is None:
        print(f"{args.file}: no metadata", file=sys.stderr)
        return 1
    print(json.dumps(meta, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
