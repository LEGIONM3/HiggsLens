from typing import Dict, List

from pydantic import BaseModel, ConfigDict

PRIMARY_FEATURES: List[str] = [
    "PRI_tau_pt", "PRI_tau_eta", "PRI_tau_phi", "PRI_lep_pt", "PRI_lep_eta", "PRI_lep_phi",
    "PRI_met", "PRI_met_phi", "PRI_met_sumet", "PRI_jet_num", "PRI_jet_leading_pt",
    "PRI_jet_leading_eta", "PRI_jet_leading_phi", "PRI_jet_subleading_pt",
    "PRI_jet_subleading_eta", "PRI_jet_subleading_phi", "PRI_jet_all_pt"
]

DERIVED_FEATURES: List[str] = [
    "DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h",
    "DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet", "DER_deltar_tau_lep",
    "DER_pt_tot", "DER_sum_pt", "DER_pt_ratio_lep_tau", "DER_met_phi_centrality",
    "DER_lep_eta_centrality"
]

METADATA_COLUMNS: List[str] = [
    "EventId", "Weight", "Label", "KaggleSet", "KaggleWeight"
]


class DatasetSummaryResponse(BaseModel):
    event_count: int
    features: List[str]
    source: str
    doi: str

    model_config = ConfigDict(frozen=True)


class PartitionSummary(BaseModel):
    partition_name: str
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
