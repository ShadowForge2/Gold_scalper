"""
Weekly retraining script — runs all model training pipelines with latest data.
Designed to be run as a cron job every weekend when markets are closed.

Usage:
    python _retrain_weekly.py                  # retrain all models (current year)
    python _retrain_weekly.py --year 2026       # include specific year
"""
import os, sys, time, argparse, logging
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WeeklyRetrain")


def run_script(name, script_path, year):
    logger.info(f"=== Starting: {name} (year={year}) ===")
    t0 = time.time()
    result = os.system(f'"{sys.executable}" "{script_path}" --year {year}')
    elapsed = time.time() - t0
    if result == 0:
        logger.info(f"=== {name} completed successfully ({elapsed:.0f}s) ===")
    else:
        logger.error(f"=== {name} FAILED (exit={result}, {elapsed:.0f}s) ===")
    return result


def main():
    parser = argparse.ArgumentParser(description="Weekly model retraining")
    parser.add_argument("--year", type=int, default=None, help="Override training end year (default: current year)")
    args = parser.parse_args()

    year = args.year or datetime.now(timezone.utc).year
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scripts = [
        ("ASP XAUUSD", os.path.join(base_dir, "_train_asp_model.py")),
        ("ASP US100", os.path.join(base_dir, "_train_asp_us100.py")),
        ("SwingQuality XAUUSD", os.path.join(base_dir, "_train_swing_xgb.py")),
        ("SwingQuality US100", os.path.join(base_dir, "_train_swing_xgb_us100.py")),
        ("Direction XAUUSD+US100", os.path.join(base_dir, "_train_direction_model.py")),
    ]

    failures = 0
    for name, path in scripts:
        if not os.path.exists(path):
            logger.warning(f"Script not found: {path}")
            failures += 1
            continue
        if run_script(name, path, year) != 0:
            failures += 1

    if failures:
        logger.error(f"Retraining finished with {failures} failure(s)")
    else:
        logger.info("All models retrained successfully")

    return failures


if __name__ == "__main__":
    sys.exit(main())
