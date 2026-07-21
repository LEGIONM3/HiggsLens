import json
import os
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from backend.app.core.config import settings


class CERNDataDownloader:
    """
    Downloads the canonical ATLAS Higgs Challenge 2014 dataset from CERN Open Data.
    Enforces atomic temporary writes, checksum/size validation, and metadata checking.
    Never silently falls back to unofficial mirrors.
    """
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.DATA_DIR
        self.raw_dir = self.data_dir / "raw"
        self.target_file = self.raw_dir / settings.DATASET_FILENAME
        self.part_file = self.raw_dir / f"{settings.DATASET_FILENAME}.part"

    def fetch_cern_metadata(self) -> Dict[str, Any]:
        """Queries CERN Open Data API for record 328 metadata."""
        url = settings.CERN_METADATA_URL
        req = urllib.request.Request(url, headers={"User-Agent": f"HiggsLens/{settings.APP_VERSION}"})
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read())
                files = data.get("metadata", {}).get("files", [])
                if not files:
                    return {}
                return files[0]
        except Exception as e:
            print(f"Warning: Could not fetch CERN API metadata ({url}): {e}")
            return {}

    def download(self, force: bool = False, dry_run: bool = False) -> Tuple[bool, str]:
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # Check cache
        if self.target_file.exists() and not force:
            if dry_run:
                return True, f"[Dry Run] Cached file exists at {self.target_file} ({self.target_file.stat().st_size} bytes)."
            return True, f"Dataset already cached at {self.target_file}. Use --force to re-download."

        url = settings.HIGGSLENS_DATASET_URL or settings.CANONICAL_DATASET_URL
        metadata = self.fetch_cern_metadata()
        expected_size = metadata.get("size")
        checksum = metadata.get("checksum")

        if dry_run:
            msg = f"[Dry Run] Would download from {url} to {self.target_file}.\n"
            if expected_size:
                msg += f"[Dry Run] CERN Record Metadata Expected Size: {expected_size} bytes, Checksum: {checksum}"
            return True, msg

        print(f"Streaming dataset from canonical source: {url}...")
        if expected_size:
            print(f"Expected file size: {expected_size / (1024*1024):.2f} MiB")

        req = urllib.request.Request(url, headers={"User-Agent": f"HiggsLens/{settings.APP_VERSION}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                # Stream into temporary partial file and compute adler32 on the fly if checksum is adler32
                adler = 1
                bytes_downloaded = 0
                with open(self.part_file, "wb") as out_f:
                    while True:
                        chunk = response.read(65536)
                        if not chunk:
                            break
                        out_f.write(chunk)
                        bytes_downloaded += len(chunk)
                        if checksum and checksum.startswith("adler32:"):
                            adler = zlib.adler32(chunk, adler)
                        if bytes_downloaded % (10 * 1024 * 1024) < 65536:
                            print(f"Downloaded {bytes_downloaded / (1024*1024):.1f} MiB...")

            print(f"Download complete ({bytes_downloaded / (1024*1024):.2f} MiB). Validating...")
            if expected_size and bytes_downloaded != expected_size:
                if self.part_file.exists():
                    self.part_file.unlink()
                return False, f"Download failure: Size mismatch. Expected {expected_size} bytes, got {bytes_downloaded}."

            if checksum and checksum.startswith("adler32:"):
                expected_adler = checksum.split(":")[1].lower()
                actual_adler = f"{adler & 0xffffffff:08x}".lower()
                if actual_adler != expected_adler:
                    if self.part_file.exists():
                        self.part_file.unlink()
                    return False, f"Checksum verification failed! Expected adler32:{expected_adler}, got adler32:{actual_adler}."
                print(f"Checksum adler32:{actual_adler} verified successfully!")

            # Atomic rename
            if self.target_file.exists():
                self.target_file.unlink()
            os.replace(self.part_file, self.target_file)
            return True, f"Successfully downloaded and verified dataset at {self.target_file}."

        except Exception as e:
            if self.part_file.exists():
                self.part_file.unlink()
            return False, f"Failed to download dataset from {url}: {str(e)}. No unofficial fallback mirrors are permitted by scientific data contract."
