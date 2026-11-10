#!/usr/bin/env python3
"""Central Neural Network AI Engine - E2E Test Suite Runner.

Executes End-to-End tests across 4 tiers:
  Tier 1: Feature Coverage (F1 to F20)
  Tier 2: Boundary & Corner Cases
  Tier 3: Cross-Feature Combinations
  Tier 4: Real-World Application Scenarios

Usage:
  python tests/e2e/test_runner.py --tier all
  python tests/e2e/test_runner.py --tier 1
  python tests/e2e/test_runner.py --tier 2
  python tests/e2e/test_runner.py --tier 3
  python tests/e2e/test_runner.py --tier 4
"""

import argparse
import os
import sys
import time
import unittest
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"

TIER_METADATA = {
    1: {
        "name": "Tier 1: Feature Coverage",
        "module": "tests.e2e.test_tier1_features",
        "description": "Core capability coverage across F1-F20 (>=5 tests per feature)",
    },
    2: {
        "name": "Tier 2: Boundary & Corner Cases",
        "module": "tests.e2e.test_tier2_boundaries",
        "description": "Extreme limits, 100% RH, calm wind, station dropouts, spikes",
    },
    3: {
        "name": "Tier 3: Feature Combinations",
        "module": "tests.e2e.test_tier3_combinations",
        "description": "Multi-module integration pipelines (Sim -> Calib -> WLS -> Forecast -> AE)",
    },
    4: {
        "name": "Tier 4: Real-World Scenarios",
        "module": "tests.e2e.test_tier4_scenarios",
        "description": "Complex Hanoi smog episodes, nocturnal inversion, fire outbreak, sensor drift",
    },
}

def load_tier_suite(tier_num: int) -> unittest.TestSuite:
    """Dynamically loads the test suite for the requested tier."""
    if tier_num not in TIER_METADATA:
        raise ValueError(f"Unknown tier: {tier_num}")
    mod_name = TIER_METADATA[tier_num]["module"]
    loader = unittest.defaultTestLoader
    try:
        suite = loader.loadTestsFromName(mod_name)
        return suite
    except Exception as e:
        print(f"{Colors.RED}[ERROR] Failed to load module {mod_name}: {e}{Colors.RESET}")
        raise

def run_tier(tier_num: int, verbose: bool = False, failfast: bool = False) -> Tuple[unittest.TestResult, float]:
    """Runs a single test tier and returns (result, duration_seconds)."""
    suite = load_tier_suite(tier_num)
    start_time = time.time()
    stream = sys.stdout if verbose else open(os.devnull, "w")
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=2 if verbose else 1,
        failfast=failfast,
    )
    result = runner.run(suite)
    duration = time.time() - start_time
    return result, duration

def format_row(cols: List[str], widths: List[int]) -> str:
    """Formats a single row with specific column widths."""
    return " | ".join(c.ljust(w) for c, w in zip(cols, widths))

def print_banner():
    """Prints beautiful header banner."""
    print(f"\n{Colors.CYAN}{'=' * 88}")
    print(f" {Colors.BOLD}CENTRAL NEURAL NETWORK AI ENGINE — END-TO-END TEST SUITE RUNNER{Colors.RESET}{Colors.CYAN}")
    print(f" Target Hardware: Raspberry Pi 5 (Quad ARM Cortex-A76 @ 2.4 GHz, 8GB LPDDR4X)")
    print(f" Standards: Category-Partition, BVA, Pairwise Combinations, Physical Scenarios")
    print(f"{'=' * 88}{Colors.RESET}\n")

def print_summary_table(results_by_tier: Dict[int, Tuple[unittest.TestResult, float]]):
    """Prints a beautiful summary table of all test tiers."""
    col_headers = ["Tier", "Description", "Total", "Pass", "Fail", "Error", "Skip", "Time (s)", "Status"]
    col_widths = [8, 38, 6, 6, 6, 6, 6, 9, 8]

    sep_line = "-+-".join("-" * w for w in col_widths)

    print(f"{Colors.BOLD}{format_row(col_headers, col_widths)}{Colors.RESET}")
    print(sep_line)

    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_errors = 0
    total_skipped = 0
    total_time = 0.0

    all_passed = True

    for tier_num, (res, duration) in sorted(results_by_tier.items()):
        name = f"Tier {tier_num}"
        desc = TIER_METADATA[tier_num]["description"][:38]
        t_count = res.testsRun
        f_count = len(res.failures)
        e_count = len(res.errors)
        s_count = len(res.skipped)
        p_count = t_count - f_count - e_count - s_count

        total_tests += t_count
        total_passed += p_count
        total_failed += f_count
        total_errors += e_count
        total_skipped += s_count
        total_time += duration

        tier_ok = (f_count == 0 and e_count == 0)
        if not tier_ok:
            all_passed = False

        status_str = f"{Colors.GREEN}PASS{Colors.RESET}" if tier_ok else f"{Colors.RED}FAIL{Colors.RESET}"

        row = [
            name,
            desc,
            str(t_count),
            f"{Colors.GREEN}{p_count}{Colors.RESET}",
            f"{Colors.RED}{f_count}{Colors.RESET}" if f_count > 0 else "0",
            f"{Colors.RED}{e_count}{Colors.RESET}" if e_count > 0 else "0",
            f"{Colors.YELLOW}{s_count}{Colors.RESET}" if s_count > 0 else "0",
            f"{duration:.3f}",
            status_str,
        ]
        print(format_row(row, col_widths))

    print(sep_line)

    summary_status = f"{Colors.GREEN}PASSED{Colors.RESET}" if all_passed else f"{Colors.RED}FAILED{Colors.RESET}"
    totals_row = [
        "TOTALS",
        "All Executed Tiers Combined",
        str(total_tests),
        f"{Colors.GREEN}{total_passed}{Colors.RESET}",
        f"{Colors.RED}{total_failed}{Colors.RESET}" if total_failed > 0 else "0",
        f"{Colors.RED}{total_errors}{Colors.RESET}" if total_errors > 0 else "0",
        f"{Colors.YELLOW}{total_skipped}{Colors.RESET}" if total_skipped > 0 else "0",
        f"{total_time:.3f}",
        summary_status,
    ]
    print(f"{Colors.BOLD}{format_row(totals_row, col_widths)}{Colors.RESET}")
    print()

    has_issues = False
    for tier_num, (res, _) in results_by_tier.items():
        if res.failures or res.errors:
            has_issues = True
            print(f"{Colors.RED}{'=' * 88}")
            print(f" FAILURES & ERRORS IN TIER {tier_num}")
            print(f"{'=' * 88}{Colors.RESET}")
            for test, err in res.failures:
                print(f"{Colors.RED}[FAIL]{Colors.RESET} {test}:")
                print(err)
            for test, err in res.errors:
                print(f"{Colors.RED}[ERROR]{Colors.RESET} {test}:")
                print(err)

    if not has_issues:
        print(f"{Colors.GREEN}{Colors.BOLD}[OK] ALL {total_tests} TEST CASES PASSED SUCCESSFULLY IN {total_time:.2f}s!{Colors.RESET}\n")
    else:
        print(f"{Colors.RED}{Colors.BOLD}[FAIL] TEST SUITE FAILED WITH {total_failed} FAILURES AND {total_errors} ERRORS.{Colors.RESET}\n")

    return all_passed

def main():
    parser = argparse.ArgumentParser(description="Central AI Engine E2E Test Suite Runner")
    parser.add_argument(
        "--tier",
        type=str,
        default="all",
        choices=["1", "2", "3", "4", "all"],
        help="Specify which tier to execute (1, 2, 3, 4, or all)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose test output per test case",
    )
    parser.add_argument(
        "-f", "--failfast",
        action="store_true",
        help="Stop on first failure or error",
    )
    args = parser.parse_args()

    print_banner()

    tiers_to_run = [1, 2, 3, 4] if args.tier == "all" else [int(args.tier)]

    results_by_tier: Dict[int, Tuple[unittest.TestResult, float]] = {}

    for t in tiers_to_run:
        meta = TIER_METADATA[t]
        print(f"{Colors.CYAN}[RUN] Executing {meta['name']}...{Colors.RESET}")
        res, duration = run_tier(t, verbose=args.verbose, failfast=args.failfast)
        results_by_tier[t] = (res, duration)

    print()
    all_passed = print_summary_table(results_by_tier)

    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
