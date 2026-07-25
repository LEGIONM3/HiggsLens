#!/usr/bin/env python
"""
HiggsLens Model Artifact Packaging CLI

Inference-only architecture: The backend service layer strictly loads pre-trained model
artifacts from versioned model registries (models/artifacts/{model_id}/).
This script invokes the artifact migration script to package candidate model artifacts.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.migrate_artifacts import migrate  # noqa: E402


def main():
    print("=" * 70)
    print("HiggsLens Model Artifact Packaging CLI (Inference-Only)")
    print("=" * 70)
    print("Packaging pre-trained model artifacts into versioned contract layout...")
    migrate()


if __name__ == "__main__":
    main()
