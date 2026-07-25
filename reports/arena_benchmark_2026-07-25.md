# HiggsLens — Official Model Arena Benchmark Report

**Session**: R004 — Arena Benchmark (GPU-first)  
**Date**: 2026-07-25  
**Commit**: `625d028`  

---

## 1. Environment

```text
Sat Jul 25 21:10:08 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.79                 Driver Version: 595.79         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|===============
```

| Item | Value |
| :--- | :--- |
| PyTorch GPU | cuda:0 (NVIDIA GeForce RTX 5070 Ti Laptop GPU, sm_120) |
| Dataset SHA-256 | `54242acf28a78ce303ea48bcf7002f0a44df08448271477e0a63331486c4f316` |
| Train split | 250,000 events |
| Val split | 100,000 events |
| Test split | 450,000 events |
| Holdout | 18,238 events — UNTOUCHED |

---

## 2. Leaderboard (sorted by Test ROC-AUC)

> Fit on train → threshold on val → final metrics on test

| Rank | Model | Device | Val AUC | Test AUC | Test AMS | Thresh | F1 | Fit | Lat/event |
| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **xgboost** | `cuda:0` GPU | 0.9082 | **0.9096** | 3.5526 | 0.8118 | 0.5816 | 0.40s | 0.0002ms |
| 2 | **lightgbm** | `gpu/cpu` GPU | 0.9072 | **0.9087** | 3.5583 | 0.8217 | 0.5637 | 0.64s | 0.0004ms |
| 3 | **mlp_torch** | `cuda:0` GPU | 0.9053 | **0.9064** | 3.3296 | 0.7623 | 0.6281 | 66.50s | 0.0002ms |
| 4 | **random_forest** | `cpu` | 0.9043 | **0.9061** | 3.4880 | 0.7623 | 0.5902 | 6.06s | 0.0010ms |
| 5 | **mlp** | `cpu` | 0.9043 | **0.9060** | 3.4216 | 0.8316 | 0.5607 | 17.81s | 0.0005ms |
| 6 | **histogram_gradient_boosting** | `cpu` | 0.8988 | **0.9003** | 3.3986 | 0.7920 | 0.5102 | 1.95s | 0.0004ms |
| 7 | **calibrated_ensemble** | `cpu` | 0.8944 | **0.8963** | 3.3628 | 0.6930 | 0.5545 | 8.89s | 0.0014ms |
| 8 | **svm_rbf** | `cpu` | 0.8705 | **0.8723** | 2.7198 | 0.7722 | 0.5144 | 11.20s | 0.2847ms |
| 9 | **logistic_regression** | `cpu` | 0.8128 | **0.8146** | 2.0590 | 0.4159 | 0.6459 | 0.46s | 0.0002ms |
| 10 | **dummy_prior** | `cpu` | 0.5000 | **0.5000** | 1.0791 | 0.0100 | 0.5092 | 0.02s | 0.0000ms |

**Non-completed models:**

| Model | Status | Reason |
| :--- | :--- | :--- |
| `quantum_kernel_svm` | skipped | missing: qiskit_machine_learning |
| `variational_quantum_classifier` | skipped | missing: pennylane |

---

## 3. Scientific Disclaimer

Pre-trained certified weights benchmarked on CERN/ATLAS open data (record 328, DOI: `10.7483/OPENDATA.ATLAS.ZBP2.M5T8`). Not CERN-validated. Educational/demonstrative purposes only.
