"""Generate paired-seed reports from the frozen experiment ledger."""

import argparse
import json
import logging
from pathlib import Path

from project_module.e_means.analysis import summarize_runs


def main() -> int:
    """Write complete or partial analysis; return 2 if strict completeness fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    target = summarize_runs([args.runs_root], args.output_root)
    summary = json.loads(target.read_text(encoding="utf-8"))
    return 2 if args.require_complete and not summary["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
