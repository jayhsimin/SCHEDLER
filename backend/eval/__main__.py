"""
CLI entry point.  Run with:
    python -m backend.eval
    python -m backend.eval --case TC-06
    python -m backend.eval --verbose
"""
import argparse
import io
import sys
from dotenv import load_dotenv
load_dotenv()

# Force UTF-8 output so Chinese chars and symbols don't crash on Windows cp950
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from .cases import CASES
from .runner import run_all, run_case, CaseResult


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def print_report(results: list[CaseResult], verbose: bool = False) -> None:
    print("\n" + "=" * 60)
    print("  EVAL REPORT")
    print("=" * 60)

    total = len(results)
    passed_count = sum(1 for r in results if r.passed)

    for r in results:
        status = _status(r.passed)
        gap_str = f"  gap={r.fairness_gap}" if r.fairness_gap is not None else ""
        print(f"\n[{status}] {r.case_id} - {r.description}{gap_str}")

        if r.error:
            print(f"  ERROR: {r.error}")
            continue

        if verbose or not r.passed:
            for chk in r.extraction_checks + r.schedule_checks:
                mark = "OK" if chk.passed else "XX"
                line = f"  [{mark}] {chk.name}"
                if not chk.passed and chk.detail:
                    line += f"\n        -> {chk.detail}"
                print(line)

    print("\n" + "=" * 60)
    pct = 100 * passed_count // total if total else 0
    print(f"  Result: {passed_count}/{total} passed ({pct}%)")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scheduling eval pipeline")
    parser.add_argument("--case", help="Run a single case by ID (e.g. TC-06)")
    parser.add_argument("--verbose", action="store_true", help="Show all checks, not just failures")
    args = parser.parse_args()

    if args.case:
        selected = [c for c in CASES if c.id == args.case]
        if not selected:
            print(f"Unknown case ID: {args.case}. Available: {[c.id for c in CASES]}")
            sys.exit(1)
        results = [run_case(selected[0])]
    else:
        results = run_all(CASES)

    print_report(results, verbose=args.verbose)

    all_passed = all(r.passed for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
