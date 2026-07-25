"""
Deterministic Dataset Preparation Pipeline for HiggsLens ML Model Arena.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.data.feature_sets import ALL_PHYSICS_FEATURES, FEATURE_SETS, get_feature_set

logger = logging.getLogger("higgslens.ml.prep_pipeline")


class DatasetPrepPipeline:
    """
    Deterministic preparation pipeline loading raw CERN collision data, coercing dtypes,
    applying configurable missing sentinel (-999.0) strategies, mapping the official KaggleSet
    column-based partition, normalizing event weights to full-dataset target luminosity,
    and generating idempotent dataset_manifest.json manifests.
    """

    KAGGLESET_MAPPING: Dict[str, str] = {
        "t": "train",
        "b": "validation",
        "v": "test",
        "u": "holdout",
    }

    def __init__(
        self,
        raw_filepath: Optional[Path] = None,
        sentinel_strategy: str = "keep-as-value",
        seed: int = 42,
        dataset_version: str = "v1",
    ):
        self.raw_filepath = raw_filepath
        self.sentinel_strategy = sentinel_strategy
        self.seed = seed
        self.dataset_version = dataset_version

        if sentinel_strategy not in ("keep-as-value", "mask", "impute"):
            raise ValueError(f"Unknown sentinel_strategy '{sentinel_strategy}'. Must be 'keep-as-value', 'mask', or 'impute'.")

    def load_and_coerce_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerces columns to exact dtypes."""
        df = df.copy()

        if "EventId" in df.columns:
            df["EventId"] = df["EventId"].astype(np.int64)

        if "Weight" in df.columns:
            df["Weight"] = df["Weight"].astype(np.float64)

        if "Label" in df.columns:
            df["Label"] = df["Label"].astype(str)

        if "KaggleSet" in df.columns:
            df["KaggleSet"] = df["KaggleSet"].astype(str)

        for col in ALL_PHYSICS_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)

        return df

    def apply_sentinel_strategy(
        self,
        splits: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Applies requested -999.0 sentinel processing strategy."""
        if self.sentinel_strategy == "keep-as-value":
            return splits

        processed = {}
        if self.sentinel_strategy == "mask":
            for name, df in splits.items():
                df_copy = df.copy()
                cols = [c for c in ALL_PHYSICS_FEATURES if c in df_copy.columns]
                df_copy[cols] = df_copy[cols].replace(-999.0, np.nan)
                processed[name] = df_copy
            return processed

        # impute strategy: compute median on training split ONLY
        train_df = splits["train"].copy()
        feature_cols = [c for c in ALL_PHYSICS_FEATURES if c in train_df.columns]
        medians = {}
        for col in feature_cols:
            valid_vals = train_df.loc[train_df[col] != -999.0, col]
            medians[col] = float(valid_vals.median()) if len(valid_vals) > 0 else 0.0

        for name, df in splits.items():
            df_copy = df.copy()
            for col in feature_cols:
                if col in df_copy.columns:
                    mask = df_copy[col] == -999.0
                    df_copy.loc[mask, col] = medians[col]
            processed[name] = df_copy

        return processed

    def renormalize_weights(
        self,
        splits: Dict[str, pd.DataFrame],
        full_df: pd.DataFrame
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, float]]]:
        """
        Renormalizes per-split event weights so signal and background weight sums in each split
        scale to match the target full-dataset luminosity scale.
        Adds 'RenormalizedWeight' column to every split DataFrame.
        """
        s_full = float(full_df.loc[full_df["Label"] == "s", "Weight"].sum())
        b_full = float(full_df.loc[full_df["Label"] == "b", "Weight"].sum())

        renorm_factors: Dict[str, Dict[str, float]] = {}
        processed_splits: Dict[str, pd.DataFrame] = {}

        for split_name, df in splits.items():
            df_copy = df.copy()
            s_split = float(df_copy.loc[df_copy["Label"] == "s", "Weight"].sum())
            b_split = float(df_copy.loc[df_copy["Label"] == "b", "Weight"].sum())

            f_s = (s_full / s_split) if s_split > 0 else 1.0
            f_b = (b_full / b_split) if b_split > 0 else 1.0

            renorm_factors[split_name] = {
                "signal_factor": f_s,
                "background_factor": f_b,
                "split_signal_weight_sum": s_split,
                "split_background_weight_sum": b_split,
                "full_signal_weight_sum": s_full,
                "full_background_weight_sum": b_full,
            }

            weights_renorm = np.zeros(len(df_copy), dtype=np.float64)
            is_signal = (df_copy["Label"] == "s").to_numpy()
            raw_w = df_copy["Weight"].to_numpy(dtype=np.float64)

            weights_renorm[is_signal] = raw_w[is_signal] * f_s
            weights_renorm[~is_signal] = raw_w[~is_signal] * f_b

            df_copy["RenormalizedWeight"] = weights_renorm
            processed_splits[split_name] = df_copy

        return processed_splits, renorm_factors

    def process(self, df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
        """Processes raw dataset into splits and computes dataset manifest."""
        df_coerced = self.load_and_coerce_dtypes(df)

        if "KaggleSet" not in df_coerced.columns:
            raise ValueError("Dataset missing 'KaggleSet' partition column.")

        # Deterministic column-based split mapping
        raw_splits = {
            "train": df_coerced[df_coerced["KaggleSet"] == "t"].copy(),
            "validation": df_coerced[df_coerced["KaggleSet"] == "b"].copy(),
            "test": df_coerced[df_coerced["KaggleSet"] == "v"].copy(),
            "holdout": df_coerced[df_coerced["KaggleSet"] == "u"].copy(),
        }

        # Deterministic row shuffle within each split using seed 42
        for name, split_df in raw_splits.items():
            if len(split_df) > 0:
                raw_splits[name] = split_df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

        splits, renorm_factors = self.renormalize_weights(raw_splits, df_coerced)
        splits = self.apply_sentinel_strategy(splits)

        row_counts = {k: len(v) for k, v in splits.items()}
        total_rows = len(df_coerced)
        split_ratios = {k: float(v / total_rows) if total_rows > 0 else 0.0 for k, v in row_counts.items()}

        manifest_data: Dict[str, Any] = {
            "dataset_version": self.dataset_version,
            "source_doi": "10.7483/OPENDATA.ATLAS.ZBP2.M5T8",
            "cern_record_id": 328,
            "canonical_split_mapping": self.KAGGLESET_MAPPING,
            "row_counts": row_counts,
            "total_events": total_rows,
            "split_ratios": split_ratios,
            "seed": self.seed,
            "sentinel_strategy": self.sentinel_strategy,
            "weight_renormalization_factors": renorm_factors,
            "feature_sets": {k: len(v["features"]) for k, v in FEATURE_SETS.items()},
        }

        return splits, manifest_data

    def run_export(
        self,
        df: pd.DataFrame,
        output_dir: Optional[Path] = None
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Executes preparation pipeline and exports processed split CSVs and dataset_manifest.json.
        Idempotent: computes deterministic content hash across exported split files.
        """
        splits, manifest = self.process(df)

        target_dir = output_dir or (Path("data") / "processed" / self.dataset_version)
        target_dir.mkdir(parents=True, exist_ok=True)

        hasher = hashlib.sha256()

        for split_name, split_df in sorted(splits.items()):
            file_path = target_dir / f"{split_name}.csv"
            # Sort columns deterministically for idempotent export
            ordered_cols = sorted(split_df.columns.tolist())
            split_df[ordered_cols].to_csv(file_path, index=False)

            # Update hash
            with open(file_path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)

        content_hash = hasher.hexdigest()
        manifest["content_hash"] = content_hash

        manifest_path = target_dir / "dataset_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Exported prepared dataset to {target_dir} (content_hash={content_hash})")
        return manifest_path, manifest
