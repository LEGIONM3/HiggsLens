# Model Card: {{ model_display_name }} (`{{ model_id }}`)

## 1. Intended Use & Scope
- **Task**: Binary classification of simulated ATLAS Higgs-to-tau-tau ($H \to \tau\tau$) collision events against background processes ($Z \to \tau\tau$, $t\bar{t}$, $W+\text{jets}$).
- **Target Audience**: Educational and scientific machine learning researchers analyzing benchmark collider datasets (`CERN Open Data Record 328`).
- **Out of Scope**: Real-time LHC trigger deployment, uncalibrated discovery claims, or generalized high-energy physics classification beyond the $8\text{ TeV}$ Higgs Challenge topology.

## 2. Dataset & Feature Set
- **Dataset Version**: ATLAS Higgs Boson Machine Learning Challenge 2014 (`DOI: 10.7483/OPENDATA.ATLAS.ZBP2.M5T8`).
- **Partitioning**: Official `KaggleSet` mapping (`t` for training, `b` for validation and threshold optimization, `v` for final test evaluation).
- **Feature Group**: `{{ feature_set }}` (`{{ num_features }}` features).

## 3. Preprocessing & Sentinel Handling
- **Missing Value Strategy**: `{{ preprocessing_strategy }}`.
- **Sentinel Handling**: `-999.0` values corresponding to physical unavailability under jet multiplicity (`PRI_jet_num`) were `{{ sentinel_handling }}`. Imputers and scalers were fit exclusively on training data.

## 4. Evaluation Metrics (Validation Set)
- **ROC-AUC**: `{{ roc_auc_mean }} ± {{ roc_auc_std }}`
- **PR-AUC**: `{{ pr_auc_mean }} ± {{ pr_auc_std }}`
- **Log Loss**: `{{ log_loss_mean }} ± {{ log_loss_std }}`
- **Balanced Accuracy**: `{{ balanced_accuracy_mean }} ± {{ balanced_accuracy_std }}`
- **F1 Score**: `{{ f1_mean }} ± {{ f1_std }}`
- **Brier Score (Calibration)**: `{{ brier_score_mean }} ± {{ brier_score_std }}`
- **Optimal Decision Threshold (Validation)**: `{{ optimal_threshold }}`
- **Approximate Median Significance (AMS)**: `{{ ams_score }}` (evaluated at $b_r = 10$)

## 5. Stability Across Seeds & Calibration
- **Seeds Evaluated**: `{{ seeds }}`
- **Stability Assessment**: `{{ stability_status }}`
- **Calibration Status**: `{{ calibration_status }}`

## 6. Ethical & Scientific Caveats
- Predictions represent statistical similarity to Monte Carlo simulation templates. High classifier confidence does not imply real-world particle existence or physical discovery.
