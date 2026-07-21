# Dataset Card: ATLAS Higgs Boson Machine Learning Challenge 2014

## Official Source & Citation
- **Source Portal**: [https://opendata.cern.ch/record/328](https://opendata.cern.ch/record/328)
- **DOI**: `10.7483/OPENDATA.ATLAS.ZBP2.M5T8`
- **Full Citation**: ATLAS collaboration (2014). *Dataset from the ATLAS Higgs Boson Machine Learning Challenge 2014*. CERN Open Data Portal. DOI: `10.7483/OPENDATA.ATLAS.ZBP2.M5T8`
- **Release License**: CC0 1.0 Universal Public Domain Dedication.

## Simulation & Physics Context
The dataset simulates proton-proton collisions at $\sqrt{s} = 8\text{ TeV}$ using full ATLAS detector modeling.
- **Signal ($s$)**: $H \to \tau^+\tau^-$ (Higgs boson mass fixed at $125\text{ GeV}$).
- **Background ($b$)**:
  - $Z \to \tau^+\tau^-$ (irreducible background with identical decay topology).
  - $t\bar{t}$ (top-pair production yielding leptons and jets).
  - $W + \text{jets}$ (where a jet is misidentified as a hadronic tau).

## Schema & Feature Groups
Total events: `818,238`. Columns: `35`.

1. **Metadata / Non-Predictive Columns (`5`)**:
   - `EventId`: Unique numerical event identifier.
   - `Weight`: Physics event weight corresponding to expected luminosity and cross-section.
   - `Label`: Ground truth string (`s` for signal, `b` for background).
   - `KaggleSet`: Partition flag (`t`: train, `b`: validation/public leaderboard, `v`: test/private leaderboard, `u`: unused/holdout).
   - `KaggleWeight`: Normalized weight used for Kaggle challenge scoring.
2. **Primary Features (`17` `PRI_*`)**:
   - Kinematics of reconstructed objects: `tau` ($p_T, \eta, \phi$), `lepton` ($p_T, \eta, \phi$), `MET` (magnitude, $\phi$, `sumet`), and jets (`PRI_jet_num`, leading and subleading jet $p_T, \eta, \phi$, `PRI_jet_all_pt`).
3. **Derived Features (`13` `DER_*`)**:
   - High-level variables constructed by physicists, such as `DER_mass_MMC` (Missing Mass Calculator estimation), `DER_mass_transverse_met_lep`, `DER_deltaeta_jet_jet`, `DER_mass_jet_jet`, `DER_pt_h`, `DER_deltar_tau_lep`, and centralities.

## Missing Values & Jet Multiplicity (`-999.0`)
A sentinel value of `-999.0` represents physical impossibility or measurement unavailability:
- When `PRI_jet_num == 0`, leading jet properties (`PRI_jet_leading_*`) and two-jet properties (`DER_*_jet_jet`, subleading jet properties) equal `-999.0`.
- When `PRI_jet_num == 1`, subleading jet properties (`PRI_jet_subleading_*`) and two-jet properties equal `-999.0`.
- `DER_mass_MMC` equals `-999.0` when the Higgs candidate topology cannot be solved by the Missing Mass Calculator algorithm.

## Leakage Prevention
To prevent scientific invalidity, no metadata column (`EventId`, `Label`, `Weight`, `KaggleSet`, `KaggleWeight`) is ever included in model feature matrices.
