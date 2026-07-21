import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from backend.app.core.config import settings
from backend.app.schemas.dataset import (
    DERIVED_FEATURES,
    METADATA_COLUMNS,
    PRIMARY_FEATURES,
    DatasetValidationReport,
    PartitionSummary,
)


class DatasetValidator:
    """
    Runs comprehensive scientific checks on the CERN Higgs challenge dataset.
    Generates structured validation reports and saves JSON/Markdown artifacts.
    """
    def __init__(self, filepath: Optional[Path] = None):
        self.filepath = filepath or settings.get_raw_dataset_path()

    def compute_fingerprint(self, df: pd.DataFrame) -> str:
        if self.filepath.exists():
            h = hashlib.sha256()
            with open(self.filepath, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            return h.hexdigest()
        # Fallback in-memory hash
        return hashlib.sha256(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()

    def validate(self, df: Optional[pd.DataFrame] = None) -> DatasetValidationReport:
        if df is None:
            if not self.filepath.exists():
                raise FileNotFoundError(f"Dataset file {self.filepath} missing.")
            df = pd.read_csv(self.filepath)

        row_count, col_count = df.shape
        warnings: List[str] = []
        errors: List[str] = []
        schema_mismatches: List[str] = []

        # Check required columns
        expected_cols = set(PRIMARY_FEATURES + DERIVED_FEATURES + METADATA_COLUMNS)
        actual_cols = set(df.columns)
        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols
        if missing_cols:
            schema_mismatches.append(f"Missing expected columns: {sorted(missing_cols)}")
            errors.append(f"Missing required columns: {sorted(missing_cols)}")
        if extra_cols:
            schema_mismatches.append(f"Unexpected extra columns: {sorted(extra_cols)}")

        # Label distribution
        label_dist = df["Label"].value_counts().to_dict() if "Label" in df.columns else {}

        # Weight summary
        weight_sum = {}
        if "Weight" in df.columns:
            weight_sum = {
                "min": float(df["Weight"].min()),
                "max": float(df["Weight"].max()),
                "mean": float(df["Weight"].mean()),
                "sum": float(df["Weight"].sum()),
                "signal_weight_sum": float(df[df["Label"] == "s"]["Weight"].sum()) if "Label" in df.columns else 0.0,
                "background_weight_sum": float(df[df["Label"] == "b"]["Weight"].sum()) if "Label" in df.columns else 0.0,
            }

        # KaggleSet distribution
        kaggleset_dist = df["KaggleSet"].value_counts().to_dict() if "KaggleSet" in df.columns else {}

        # Partition summaries
        partition_summaries: List[PartitionSummary] = []
        if "KaggleSet" in df.columns and "Label" in df.columns:
            mapping = {"t": "train", "b": "validation", "v": "test", "u": "holdout"}
            for code, name in mapping.items():
                part_df = df[df["KaggleSet"] == code]
                sc = int((part_df["Label"] == "s").sum())
                bc = int((part_df["Label"] == "b").sum())
                tot = len(part_df)
                partition_summaries.append(PartitionSummary(
                    partition_name=name,
                    event_count=tot,
                    signal_count=sc,
                    background_count=bc,
                    signal_ratio=float(sc / tot) if tot > 0 else 0.0
                ))

        # Missing / sentinel count (-999.0) by feature
        sentinel_counts: Dict[str, int] = {}
        infinite_count = 0
        for col in PRIMARY_FEATURES + DERIVED_FEATURES:
            if col in df.columns:
                sc = int((df[col] == -999.0).sum())
                sentinel_counts[col] = sc
                inf_c = int(np.isinf(df[col]).sum())
                infinite_count += inf_c

        # Jet multiplicity distribution
        jet_dist = df["PRI_jet_num"].value_counts().to_dict() if "PRI_jet_num" in df.columns else {}
        # Convert keys to str for JSON serialization
        jet_dist = {str(k): int(v) for k, v in jet_dist.items()}

        # Duplicate EventId count
        dup_ids = int(df["EventId"].duplicated().sum()) if "EventId" in df.columns else 0
        if dup_ids > 0:
            errors.append(f"Found {dup_ids} duplicate EventIds in dataset.")

        if infinite_count > 0:
            errors.append(f"Found {infinite_count} infinite values in numeric features.")

        # Jet multiplicity rule checks
        if "PRI_jet_num" in df.columns and "PRI_jet_leading_pt" in df.columns:
            jets_0 = df[df["PRI_jet_num"] == 0]
            if not (jets_0["PRI_jet_leading_pt"] == -999.0).all():
                warnings.append("Some events with PRI_jet_num==0 do not have -999.0 for PRI_jet_leading_pt.")

        is_valid = len(errors) == 0
        fingerprint = self.compute_fingerprint(df)

        report = DatasetValidationReport(
            is_valid=is_valid,
            row_count=row_count,
            column_count=col_count,
            schema_mismatches=schema_mismatches,
            label_distribution=label_dist,
            weight_summary=weight_sum,
            kaggleset_distribution=kaggleset_dist,
            partition_summaries=partition_summaries,
            missing_sentinel_count_by_feature=sentinel_counts,
            jet_multiplicity_distribution=jet_dist,
            duplicate_event_id_count=dup_ids,
            infinite_values_count=infinite_count,
            data_fingerprint=fingerprint,
            warnings=warnings,
            errors=errors
        )

        self.save_artifacts(report)
        return report

    def save_artifacts(self, report: DatasetValidationReport) -> None:
        metrics_dir = settings.ARTIFACTS_DIR / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        json_path = metrics_dir / "dataset_validation_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        md_path = metrics_dir / "dataset_validation_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Dataset Validation Report\n\n")
            f.write(f"- **Valid**: `{report.is_valid}`\n")
            f.write(f"- **Rows**: `{report.row_count:,}` | **Columns**: `{report.column_count}`\n")
            f.write(f"- **Fingerprint (SHA256)**: `{report.data_fingerprint}`\n\n")
            f.write("## Partitions (`KaggleSet` Mapping)\n")
            f.write("| Partition | Events | Signal | Background | Signal Ratio |\n| :--- | :--- | :--- | :--- | :--- |\n")
            for p in report.partition_summaries:
                f.write(f"| **{p.partition_name.capitalize()}** | {p.event_count:,} | {p.signal_count:,} | {p.background_count:,} | {p.signal_ratio:.2%} |\n")
            f.write("\n## Sentinel Frequency (`-999.0`)\n")
            for feat, sc in report.missing_sentinel_count_by_feature.items():
                if sc > 0:
                    f.write(f"- `{feat}`: {sc:,} ({sc / report.row_count:.2%})\n")
            if report.errors:
                f.write("\n## Errors\n")
                for err in report.errors:
                    f.write(f"- **ERROR**: {err}\n")
            if report.warnings:
                f.write("\n## Warnings\n")
                for w in report.warnings:
                    f.write(f"- **WARNING**: {w}\n")
