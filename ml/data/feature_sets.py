"""
Declarative Feature Set Definitions for HiggsLens ML Model Arena.
"""

from typing import Dict, List, TypedDict


class FeatureSetSpec(TypedDict):
    name: str
    description: str
    feature_count: int
    features: List[str]


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

ALL_PHYSICS_FEATURES: List[str] = DERIVED_FEATURES + PRIMARY_FEATURES

FEATURE_SETS: Dict[str, FeatureSetSpec] = {
    "all_physics": {
        "name": "all_physics",
        "description": "Full set of 30 physical features (17 primary detector + 13 derived physics variables).",
        "feature_count": 30,
        "features": ALL_PHYSICS_FEATURES,
    },
    "pri_only": {
        "name": "pri_only",
        "description": "Primary detector-level reconstructed measurements (17 PRI_* variables).",
        "feature_count": 17,
        "features": PRIMARY_FEATURES,
    },
    "der_only": {
        "name": "der_only",
        "description": "Physicist-engineered derived features (13 DER_* variables).",
        "feature_count": 13,
        "features": DERIVED_FEATURES,
    },
}


def get_feature_set(name: str = "all_physics") -> List[str]:
    """Returns list of feature names for requested feature set."""
    if name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{name}'. Available: {list(FEATURE_SETS.keys())}")
    return FEATURE_SETS[name]["features"]
