#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.core.config import settings
from backend.app.data.validator import DatasetValidator


def main():
    parser = argparse.ArgumentParser(description="Validate ATLAS Higgs Challenge dataset schema, sentinels, and partitions.")
    parser.add_argument("--file", type=str, help="Path to raw dataset file (defaults to data/raw/atlas-higgs-challenge-2014-v2.csv.gz)")
    args = parser.parse_args()

    filepath = Path(args.file) if args.file else settings.get_raw_dataset_path()
    try:
        validator = DatasetValidator(filepath)
        report = validator.validate()
        print("\n--- Dataset Validation Completed ---")
        print(f"Valid: {report.is_valid}")
        print(f"Rows: {report.row_count:,} | Columns: {report.column_count}")
        print(f"Fingerprint (SHA256): {report.data_fingerprint}")
        if report.errors:
            print("\nErrors encountered:")
            for err in report.errors:
                print(f"  [!] {err}")
            sys.exit(1)
        else:
            print("\nAll scientific checks passed!")
            sys.exit(0)
    except Exception as e:
        print(f"Fatal error during validation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
