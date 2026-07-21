from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from backend.app.core.config import settings


class DatasetPipeline:
    """
    Loads raw CERN dataset, applies partition splitting (`official` KaggleSet vs `stratified_random`),
    and verifies zero EventId overlap across partitions.
    """
    def __init__(self, filepath: Optional[Path] = None):
        self.filepath = filepath or settings.get_raw_dataset_path()
        self._df: Optional[pd.DataFrame] = None

    def load_raw(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        if not self.filepath.exists():
            raise FileNotFoundError(f"Dataset not found at {self.filepath}. Run scripts/download_dataset.py first.")
        self._df = pd.read_csv(self.filepath)
        return self._df

    def get_partition_splits(
        self,
        strategy: str = "official_partition",
        seed: int = 42,
        sample_size: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Returns dictionary of DataFrames:
        - 'train' (t)
        - 'validation' (b)
        - 'test' (v)
        - 'holdout' (u)
        
        If sample_size is provided (e.g. for fast_mode), takes a stratified sample of the requested partitions.
        Never silently switches strategy.
        """
        df = self.load_raw()

        if strategy == "official_partition":
            if "KaggleSet" not in df.columns:
                raise ValueError("Official partition strategy requires 'KaggleSet' column, but it is missing.")

            partitions = {
                "train": df[df["KaggleSet"] == "t"].copy(),
                "validation": df[df["KaggleSet"] == "b"].copy(),
                "test": df[df["KaggleSet"] == "v"].copy(),
                "holdout": df[df["KaggleSet"] == "u"].copy(),
            }
        elif strategy == "stratified_random":
            # Explicit fallback strategy when selected
            from sklearn.model_selection import train_test_split
            train_df, rest_df = train_test_split(
                df, test_size=0.4, random_state=seed, stratify=df["Label"]
            )
            val_df, test_holdout_df = train_test_split(
                rest_df, test_size=0.5, random_state=seed, stratify=rest_df["Label"]
            )
            test_df, holdout_df = train_test_split(
                test_holdout_df, test_size=0.5, random_state=seed, stratify=test_holdout_df["Label"]
            )
            partitions = {
                "train": train_df.copy(),
                "validation": val_df.copy(),
                "test": test_df.copy(),
                "holdout": holdout_df.copy(),
            }
        else:
            raise ValueError(f"Unknown split strategy '{strategy}'. Must be 'official_partition' or 'stratified_random'.")

        # Verify no EventId overlap
        self.verify_no_overlap(partitions)

        # If sample_size requested (e.g. fast_mode bounded sample), sample each active partition stratifying by Label
        if sample_size and sample_size < len(df):
            for p_name, p_df in partitions.items():
                if len(p_df) > 0 and p_name != "holdout":  # do not sample holdout for experiments
                    # Proportional sample size based on original partition size
                    p_sample_size = min(len(p_df), max(100, int(sample_size * (len(p_df) / len(df)))))
                    from sklearn.model_selection import train_test_split
                    _, sampled_p = train_test_split(
                        p_df, test_size=p_sample_size, random_state=seed, stratify=p_df["Label"]
                    )
                    partitions[p_name] = sampled_p.copy()

        return partitions

    @staticmethod
    def verify_no_overlap(partitions: Dict[str, pd.DataFrame]) -> None:
        """Verifies that EventIds never overlap across partitions."""
        names = list(partitions.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                p1, p2 = names[i], names[j]
                df1, df2 = partitions[p1], partitions[p2]
                if "EventId" in df1.columns and "EventId" in df2.columns:
                    s1 = set(df1["EventId"])
                    s2 = set(df2["EventId"])
                    overlap = s1.intersection(s2)
                    if overlap:
                        raise RuntimeError(f"Data Leakage Error: EventId overlap detected between {p1} and {p2} ({len(overlap)} shared IDs).")

    @staticmethod
    def encode_labels(df: pd.DataFrame) -> np.ndarray:
        """Encodes string Label ('s' -> 1, 'b' -> 0). Preserves raw Label column in DataFrame."""
        return (df["Label"] == "s").astype(np.int32).to_numpy()
