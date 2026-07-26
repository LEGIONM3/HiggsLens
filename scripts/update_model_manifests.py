"""
One-time offline script to populate training_run_origin, subsample_notes, and dataset_provenance
keys in models/artifacts/*/manifest.json files.
Serving endpoints read these fields directly from manifest files (read-only).
"""

import json
from pathlib import Path

MANIFEST_EXTENSIONS = {
    "xgboost": {
        "training_run_origin": "R005 (GPU Sweep)",
        "subsample_notes": "Full 250k train set",
        "device": "cuda:0 (RTX 5070 Ti)",
    },
    "lightgbm": {
        "training_run_origin": "R005 (GPU Sweep)",
        "subsample_notes": "Full 250k train set",
        "device": "cuda:0 (RTX 5070 Ti)",
    },
    "histogram_gradient_boosting": {
        "training_run_origin": "R004 (Baseline)",
        "subsample_notes": "Full 250k train set",
        "device": "CPU (Parallel)",
    },
    "random_forest": {
        "training_run_origin": "R004 (Baseline)",
        "subsample_notes": "Full 250k train set",
        "device": "CPU (Parallel)",
    },
    "mlp_torch": {
        "training_run_origin": "R005 (GPU Sweep)",
        "subsample_notes": "Full 250k train set",
        "device": "cuda:0 (RTX 5070 Ti)",
    },
    "mlp": {
        "training_run_origin": "R004 (Baseline)",
        "subsample_notes": "Full 250k train set",
        "device": "CPU (scikit-learn)",
    },
    "svm_rbf": {
        "training_run_origin": "R005 (GPU Sweep)",
        "subsample_notes": "50k fast subsample",
        "device": "cuda:0 (cuML)",
    },
    "calibrated_ensemble": {
        "training_run_origin": "R004 (Baseline)",
        "subsample_notes": "Calibrated blend",
        "device": "CPU (scikit-learn)",
    },
    "logistic_regression": {
        "training_run_origin": "R004 (Baseline)",
        "subsample_notes": "Full 250k train set",
        "device": "CPU (scikit-learn)",
    },
    "dummy_prior": {
        "training_run_origin": "R004 (Baseline)",
        "subsample_notes": "Prior class distribution",
        "device": "CPU",
    },
    "variational_quantum_classifier": {
        "training_run_origin": "R005 (QML Benchmark)",
        "subsample_notes": "100-event budget (PennyLane)",
        "device": "CPU (lightning.qubit)",
    },
    "quantum_kernel_svm": {
        "training_run_origin": "R005 (QML Benchmark)",
        "subsample_notes": "100-event budget (Qiskit)",
        "device": "CPU (qiskit.aer)",
    },
}

COMMON_PROVENANCE = "ATLAS open data (record 328, DOI 10.7483/OPENDATA.ATLAS.ZBP2.M5T8) — official ATLAS simulated events, classified by certified pre-trained models."

def update_manifests():
    artifacts_dir = Path(__file__).resolve().parent.parent / "models" / "artifacts"
    for model_id, meta in MANIFEST_EXTENSIONS.items():
        manifest_path = artifacts_dir / model_id / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            manifest["training_run_origin"] = meta["training_run_origin"]
            manifest["subsample_notes"] = meta["subsample_notes"]
            manifest["device"] = meta["device"]
            manifest["dataset_provenance"] = COMMON_PROVENANCE
            
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Updated {model_id} manifest.json")
        else:
            print(f"WARNING: Manifest missing for {model_id}")

if __name__ == "__main__":
    update_manifests()
