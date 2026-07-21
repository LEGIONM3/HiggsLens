from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from backend.app.schemas.dataset import DERIVED_FEATURES, PRIMARY_FEATURES


@pytest.fixture
def mock_raw_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 200
    data: Dict[str, Any] = {
        "EventId": np.arange(100000, 100000 + n),
        "Label": np.random.choice(["s", "b"], size=n, p=[0.34, 0.66]),
        "Weight": np.random.uniform(0.001, 5.0, size=n),
        "KaggleSet": np.random.choice(["t", "b", "v", "u"], size=n, p=[0.5, 0.2, 0.2, 0.1]),
        "PRI_tau_pt": np.random.uniform(20.0, 150.0, size=n),
        "PRI_tau_eta": np.random.uniform(-2.5, 2.5, size=n),
        "PRI_tau_phi": np.random.uniform(-3.14, 3.14, size=n),
        "PRI_lep_pt": np.random.uniform(20.0, 150.0, size=n),
        "PRI_lep_eta": np.random.uniform(-2.5, 2.5, size=n),
        "PRI_lep_phi": np.random.uniform(-3.14, 3.14, size=n),
        "PRI_met": np.random.uniform(10.0, 200.0, size=n),
        "PRI_met_phi": np.random.uniform(-3.14, 3.14, size=n),
        "PRI_met_sumet": np.random.uniform(100.0, 800.0, size=n),
        "PRI_jet_num": np.random.choice([0, 1, 2, 3], size=n),
    }

    # Add remaining features with some -999.0 sentinels
    for col in PRIMARY_FEATURES + DERIVED_FEATURES:
        if col not in data:
            data[col] = np.random.uniform(0.0, 100.0, size=n)

    df = pd.DataFrame(data)

    # Enforce physical jet rules on sentinels
    mask_0 = df["PRI_jet_num"] == 0
    df.loc[mask_0, ["PRI_jet_leading_pt", "PRI_jet_leading_eta", "PRI_jet_leading_phi", "DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet", "DER_lep_eta_centrality"]] = -999.0

    mask_1 = df["PRI_jet_num"] <= 1
    df.loc[mask_1, ["PRI_jet_subleading_pt", "PRI_jet_subleading_eta", "PRI_jet_subleading_phi", "DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet", "DER_lep_eta_centrality"]] = -999.0

    return df
