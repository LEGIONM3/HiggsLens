# HiggsLens Terminology & Definitions

- **Signal-like**: A collision event whose kinematic signature ($p_T, \eta, \phi$, invariant masses, centralities) exhibits high statistical probability under a model trained on simulated $H \to \tau\tau$ processes.
- **Background-like**: An event matching dominant background process signatures, primarily $Z \to \tau\tau$, top-quark pairs ($t\bar{t}$), or $W+\text{jets}$.
- **Higgs Candidate**: A reconstructed event topology consisting of one hadronic tau, one light lepton ($e$ or $\mu$), and missing transverse energy ($\text{MET}$) evaluated as a potential Higgs boson decay.
- **Event Weight (`Weight`)**: The physics weight assigned to a simulated event so that weighted sums reflect expected physical event counts at a target integrated luminosity.
- **Primary Feature (`PRI_*`)**: A direct physical measurement or reconstructed object property from the ATLAS detector (e.g., `PRI_tau_pt`, `PRI_met`).
- **Derived Feature (`DER_*`)**: A composite, physicist-engineered variable combining multiple primary measurements (e.g., `DER_mass_MMC`, `DER_deltaeta_jet_jet`).
- **Invariant Mass**: The relativistic invariant mass computed from energy and momentum vectors ($M^2 = E^2 - |\mathbf{p}|^2$). In $H \to \tau\tau$, escaping neutrinos make direct invariant mass calculation impossible without specialized algorithms like the Missing Mass Calculator (`MMC`).
- **Approximate Median Significance (AMS)**: The official evaluation objective of the ATLAS Higgs Challenge, computing statistical significance of a signal selection:
  $$\text{AMS} = \sqrt{2 \left( (s + b + b_r) \ln\left(1 + \frac{s}{b + b_r}\right) - s \right)}$$
  where $s$ and $b$ are sum of `Weight` for true signal and true background events classified as positive, and $b_r = 10$ is a regularizing term preventing high variance when $b$ is small.
- **Probability Calibration**: Ensuring that when a classifier predicts a signal probability $P(s) = p$, exactly $p \times 100\%$ of those events are true signal events.
