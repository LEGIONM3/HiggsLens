#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

# Ensure root directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.data.downloader import CERNDataDownloader


def main():
    parser = argparse.ArgumentParser(description="Download ATLAS Higgs Challenge 2014 dataset from CERN Open Data.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if cached file exists.")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without writing files.")
    args = parser.parse_args()

    downloader = CERNDataDownloader()
    success, message = downloader.download(force=args.force, dry_run=args.dry_run)
    print(message)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
