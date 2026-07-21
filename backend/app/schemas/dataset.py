from typing import Dict, List, Optional

from pydantic import BaseModel

PRIMARY_FEATURES = [
    "PRI_tau_pt", "PRI_tau_eta", "PRI_tau_phi",
    "PRI_lep_pt", "PRI_lep_eta", "PRI_lep_phi",
    "PRI_met", "PRI_met_phi", "PRI_met_sumet",
    "PRI_jet_num",
    "PRI_jet_leading_pt", "PRI_jet_leading_eta", "PRI_jet_leading_phi",
    "PRI_jet_subleading_pt", "PRI_jet_subleading_eta", "PRI_jet_subleading_phi",
    "PRI_jet_all_pt"
]

DERIVED_FEATURES = [
    "DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h",
    "DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet",
    "DER_deltar_tau_lep", "DER_pt_tot", "DER_sum_pt", "DER_pt_ratio_lep_tau",
    "DER_met_phi_centrality", "DER_lep_eta_centrality"
]

METADATA_COLUMNS = [
    "EventId", "Weight", "Label", "KaggleSet", "KaggleWeight"
]


class ColumnSchema(BaseModel):
    name: str
    category: str  # "primary", "derived", "metadata"
    data_type: str
    description: Optional[str] = None
    sentinel_count: int = 0
    missing_percentage: float = 0.0


class DatasetSchemaResponse(BaseModel):
    record_id: int = 328
    doi: str = "10.7483/OPENDATA.ATLAS.ZBP2.M5T8"
    total_expected_columns: int = 35
    primary_features: List[str] = PRIMARY_FEATURES
    derived_features: List[str] = DERIVED_FEATURES
    metadata_columns: List[str] = METADATA_COLUMNS
    leakage_rules: List[str] = [
        "Never use EventId, Label, Weight, KaggleSet, or KaggleWeight as classifier inputs.",
        "Use Weight only for evaluation or explicitly configured weighted training.",
        "Preserve raw labels and encode target separately."
    ]


class PartitionSummary(BaseModel):
    partition_name: str  # train (t), validation (b), test (v), holdout (u)
    event_count: int
    signal_count: int
    background_count: int
    signal_ratio: float


class DatasetValidationReport(BaseModel):
    is_valid: bool
    row_count: int
    column_count: int
    schema_mismatches: List[str]
    label_distribution: Dict[str, int]
    weight_summary: Dict[str, float]
    kaggleset_distribution: Dict[str, int]
    partition_summaries: List[PartitionSummary]
    missing_sentinel_count_by_feature: Dict[str, int]
    jet_multiplicity_distribution: Dict[str, int]
    duplicate_event_id_count: int
    infinite_values_count: int
    data_fingerprint: str
    warnings: List[str]
    errors: List[str]


class DatasetStatusResponse(BaseModel):
    exists: bool
    filepath: str
    file_size_bytes: int
    record_id: int
    doi: str
    last_modified: Optional[str] = None
    validation_report: Optional[DatasetValidationReport] = None
