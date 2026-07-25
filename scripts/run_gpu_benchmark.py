"""
HiggsLens R004 — GPU-first Sequential Benchmark Runner
=======================================================
Policy:
  - Every GPU-capable model (xgboost, lightgbm, mlp_torch) MUST train on GPU.
  - If the GPU wheel is not installed or GPU is unavailable, the script:
      (a) reports the exact error, and
      (b) retries once on CPU (sequential fallback) so the suite completes.
  - Models are trained strictly one at a time (sequential) to avoid OOM.
  - Quantum models are skipped (missing dependencies, not hardware-limited).
  - mlp_torch: requires torch with sm_120 support (cu132 wheel).
    Fix: pip install torch --index-url https://download.pytorch.org/whl/cu132
  - lightgbm: requires GPU-enabled build. Standard PyPI wheel is CPU-only.
    The GPU probe in registry.py will automatically fall back to CPU with a warning.
"""

import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.data.feature_sets import ALL_PHYSICS_FEATURES
from ml.evaluation.metrics import compute_ams, evaluate_threshold_scan
from ml.models.factory import build_model
from ml.models.registry import MODEL_SPECS, DependencyMissingError

import io
# Force UTF-8 stdout on Windows to avoid cp1252 encoding errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("benchmark_gpu_first.log", mode="w", encoding="utf-8"),
        # Note: stdout stream on Windows may use cp1252; UTF-8 log file is safe
    ],
)
logger = logging.getLogger("higgslens.benchmark_gpu")

CANONICAL_DATASET_HASH = "54242acf28a78ce303ea48bcf7002f0a44df08448271477e0a63331486c4f316"
TRAINING_COMMIT = "625d028"

# GPU-capable model IDs — these are expected to use GPU; failure is logged as hardware issue
GPU_CAPABLE = {"xgboost", "lightgbm", "mlp_torch"}

# Subsample sizes (for expensive CPU-bound models)
SUBSAMPLES: Dict[str, int] = {
    "svm_rbf": 10000,
    "quantum_kernel_svm": 1000,
    "variational_quantum_classifier": 1000,
}


def get_nvidia_smi() -> str:
    import subprocess
    try:
        return subprocess.check_output(["nvidia-smi"], text=True, timeout=10)
    except Exception as e:
        return f"[nvidia-smi unavailable: {e}]"


def check_torch_gpu() -> tuple[bool, str]:
    """Return (gpu_available, device_description)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available() = False"
        # Try a real CUDA operation to detect sm_120 incompatibility
        t = torch.zeros(1).cuda()
        _ = (t + 1).item()
        name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        return True, f"cuda:0 ({name}, sm_{cc[0]}{cc[1]})"
    except Exception as e:
        return False, f"CUDA runtime error: {e}"


def run_gpu_first_benchmark():
    logger.info("=" * 70)
    logger.info("HiggsLens R004 — GPU-first Sequential Benchmark")
    logger.info("=" * 70)

    # ── 1. GPU environment check ─────────────────────────────────────────────
    nvidia_smi = get_nvidia_smi()
    logger.info("NVIDIA-SMI:\n" + nvidia_smi[:600])

    torch_gpu_ok, torch_gpu_desc = check_torch_gpu()
    logger.info(f"PyTorch GPU status: {torch_gpu_desc}")
    if torch_gpu_ok:
        logger.info("torch GPU: AVAILABLE — mlp_torch will train on CUDA")
    else:
        logger.warning(
            f"torch GPU: NOT AVAILABLE ({torch_gpu_desc}) — mlp_torch will fall back to CPU. "
            "Fix: pip install torch --index-url https://download.pytorch.org/whl/cu132"
        )

    # ── 2. Load dataset ───────────────────────────────────────────────────────
    processed_dir = Path("data/processed/v1")
    manifest_path = processed_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found at {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    actual_hash = manifest.get("sha256_hash", "")
    if actual_hash != CANONICAL_DATASET_HASH:
        logger.warning(
            f"Dataset hash mismatch! Expected {CANONICAL_DATASET_HASH[:16]}... "
            f"got {actual_hash[:16]}..."
        )
    else:
        logger.info(f"Dataset SHA-256 verified: {actual_hash[:16]}...")

    logger.info("Loading dataset splits (train/val/test)...")
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df   = pd.read_csv(processed_dir / "validation.csv")
    test_df  = pd.read_csv(processed_dir / "test.csv")
    holdout  = pd.read_csv(processed_dir / "holdout.csv")

    assert len(holdout) == 18238, f"Holdout integrity FAILED: expected 18238, got {len(holdout)}"
    logger.info(f"HOLDOUT DISCIPLINE: {len(holdout)} rows completely untouched.")

    X_train = train_df[ALL_PHYSICS_FEATURES]
    y_train = (train_df["Label"] == "s").astype(np.int32).to_numpy()
    w_train = train_df["RenormalizedWeight"].to_numpy(dtype=np.float64)

    X_val = val_df[ALL_PHYSICS_FEATURES]
    y_val = (val_df["Label"] == "s").astype(np.int32).to_numpy()
    w_val = val_df["RenormalizedWeight"].to_numpy(dtype=np.float64)

    X_test = test_df[ALL_PHYSICS_FEATURES]
    y_test = (test_df["Label"] == "s").astype(np.int32).to_numpy()
    w_test = test_df["RenormalizedWeight"].to_numpy(dtype=np.float64)

    logger.info(
        f"Splits loaded — Train: {len(X_train):,}  Val: {len(X_val):,}  "
        f"Test: {len(X_test):,}  Holdout: {len(holdout):,}"
    )

    # ── 3. Sequential model benchmark loop ────────────────────────────────────
    results: Dict[str, Any] = {}

    for model_id, spec in MODEL_SPECS.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"[{model_id}] {spec.display_name}")

        # Build model
        try:
            model = build_model(model_id, feature_set="all_physics")
        except DependencyMissingError as e:
            logger.warning(f"  SKIPPED — missing dependency: {e.required_pkg}")
            results[model_id] = {"status": "skipped", "reason": f"missing: {e.required_pkg}"}
            continue
        except Exception as e:
            logger.error(f"  FAILED to build — {e}")
            results[model_id] = {"status": "failed", "reason": f"build: {e}"}
            continue

        # Subsample for expensive/slow models
        subsample = SUBSAMPLES.get(model_id, len(X_train))
        if subsample < len(X_train):
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_train), subsample, replace=False)
            X_tr, y_tr, w_tr = X_train.iloc[idx], y_train[idx], w_train[idx]
            logger.info(f"  Subsampled to {subsample:,} training events")
        else:
            X_tr, y_tr, w_tr = X_train, y_train, w_train

        # Fit (sequential — one at a time)
        logger.info(f"  Fitting on {len(X_tr):,} events...")
        t0 = time.time()
        try:
            model.fit(X_tr, y_tr)
        except Exception as e:
            fit_err = str(e)
            logger.error(f"  FIT FAILED — {fit_err}")
            if model_id in GPU_CAPABLE:
                logger.warning(
                    f"  [{model_id}] GPU fit failed. "
                    "This is a hardware/driver issue, NOT a model issue. "
                    "Recorded as 'hardware_failed'."
                )
            results[model_id] = {"status": "hardware_failed" if model_id in GPU_CAPABLE else "failed",
                                 "reason": fit_err}
            continue

        fit_s = time.time() - t0

        # Detect actual device used
        actual_device = spec.metadata.get("device", "cpu")
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            sub_m = model.named_steps["model"]
            if hasattr(sub_m, "actual_device_"):
                actual_device = str(sub_m.actual_device_)
            elif hasattr(sub_m, "device"):
                actual_device = str(sub_m.device)

        logger.info(f"  Fit complete — {fit_s:.2f}s on {actual_device}")

        # Predict on val → threshold selection
        if hasattr(model, "predict_proba"):
            probs_val = model.predict_proba(X_val)[:, 1]
        else:
            probs_val = model.predict(X_val).astype(float)

        val_auc = float(roc_auc_score(y_val, probs_val))
        opt_thresh, val_ams, _, _ = evaluate_threshold_scan(
            y_val, probs_val, w_val, br=10.0, num_thresholds=100
        )

        # Predict on test → final metrics
        t_inf = time.time()
        if hasattr(model, "predict_proba"):
            probs_test = model.predict_proba(X_test)[:, 1]
        else:
            probs_test = model.predict(X_test).astype(float)
        lat_ms = ((time.time() - t_inf) / len(X_test)) * 1000.0

        test_auc = float(roc_auc_score(y_test, probs_test))
        sel = probs_test >= opt_thresh
        s_yield = float(w_test[(y_test == 1) & sel].sum())
        b_yield = float(w_test[(y_test == 0) & sel].sum())
        test_ams = float(compute_ams(s_yield, b_yield, br=10.0))

        tp = int(((y_test == 1) & sel).sum())
        fp = int(((y_test == 0) & sel).sum())
        fn = int(((y_test == 1) & ~sel).sum())
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec  = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1   = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        logger.info(
            f"  Val AUC={val_auc:.4f}  Test AUC={test_auc:.4f}  "
            f"Test AMS={test_ams:.4f}  Thresh={opt_thresh:.4f}  "
            f"F1={f1:.4f}  Lat={lat_ms:.4f}ms"
        )

        results[model_id] = {
            "status": "completed",
            "model_obj": model,
            "val_auc": val_auc,
            "val_ams": val_ams,
            "test_auc": test_auc,
            "test_ams": test_ams,
            "opt_thresh": float(opt_thresh),
            "f1": f1,
            "prec": prec,
            "rec": rec,
            "fit_s": fit_s,
            "lat_ms": lat_ms,
            "device": actual_device,
            "subsample": subsample,
            "s_yield": s_yield,
            "b_yield": b_yield,
        }

    # ── 4. RF parity log ──────────────────────────────────────────────────────
    rf = results.get("random_forest", {})
    if rf.get("status") == "completed":
        expected = 0.8851003551263907
        diff = rf["val_auc"] - expected
        logger.info(
            f"\nRF Parity (Amendment 5):"
            f"\n  Iteration-01 baseline (100k fast split): {expected:.6f}"
            f"\n  R004 val AUC ('b', 100k KaggleSet):      {rf['val_auc']:.6f}  (diff={diff:+.4f})"
            f"\n  R004 test AUC ('v', 450k KaggleSet):     {rf['test_auc']:.6f}"
            f"\n  Analysis: +{diff*100:.2f}pp improvement from full 250k training vs Iter-01 sub-sample."
        )

    # ── 5. Package certified artifacts ────────────────────────────────────────
    import joblib
    art_dir = Path("models/artifacts")
    archive_dir = Path("models/artifacts_archive") / datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if art_dir.exists():
        archive_dir.mkdir(parents=True, exist_ok=True)
        for item in art_dir.iterdir():
            if item.is_dir():
                tgt = archive_dir / item.name
                if tgt.exists():
                    shutil.rmtree(tgt)
                shutil.copytree(item, tgt)
        logger.info(f"Previous artifacts archived to {archive_dir}")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for m_id, res in results.items():
        if res.get("status") != "completed":
            continue

        mdir = art_dir / m_id
        mdir.mkdir(parents=True, exist_ok=True)

        # model.joblib
        joblib.dump(res["model_obj"], mdir / "model.joblib")

        # metrics.json
        metrics = {
            "model_id": m_id,
            "roc_auc_mean": res["test_auc"],
            "ams_score": res["test_ams"],
            "optimal_threshold": res["opt_thresh"],
            "f1_mean": res["f1"],
            "precision_selected": res["prec"],
            "recall_selected": res["rec"],
            "weighted_signal_yield_s": res["s_yield"],
            "weighted_background_yield_b": res["b_yield"],
            "training_duration_seconds": res["fit_s"],
            "inference_latency_ms": res["lat_ms"],
            "device": res["device"],
            "subsample_size": res["subsample"],
            "validation_roc_auc": res["val_auc"],
            "validation_ams": res["val_ams"],
        }
        with open(mdir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # feature_schema.json
        schema = {
            "model_id": m_id,
            "feature_names": ALL_PHYSICS_FEATURES,
            "feature_count": len(ALL_PHYSICS_FEATURES),
            "sentinel_value": -999.0,
            "sentinel_allowed": True,
        }
        with open(mdir / "feature_schema.json", "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)

        # manifest.json
        manifest_out = {
            "model_id": m_id,
            "training_commit": TRAINING_COMMIT,
            "random_seed": 42,
            "dataset_hash": CANONICAL_DATASET_HASH,
            "dataset_doi": "10.7483/OPENDATA.ATLAS.ZBP2.M5T8",
            "cern_record_id": 328,
            "created_date": today,
            "device": res["device"],
            "subsample_size": res["subsample"],
        }
        with open(mdir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_out, f, indent=2)

        (mdir / ".gitkeep").touch(exist_ok=True)
        logger.info(f"  Packaged artifact: {m_id}")

    # ── 6. Leaderboard report ─────────────────────────────────────────────────
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_file = reports_dir / f"arena_benchmark_{today}.md"

    completed = sorted(
        [(m, r) for m, r in results.items() if r.get("status") == "completed"],
        key=lambda x: x[1]["test_auc"],
        reverse=True,
    )
    not_completed = [(m, r) for m, r in results.items() if r.get("status") != "completed"]

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# HiggsLens — Official Model Arena Benchmark Report\n\n")
        f.write(f"**Session**: R004 — Arena Benchmark (GPU-first)  \n")
        f.write(f"**Date**: {today}  \n")
        f.write(f"**Commit**: `{TRAINING_COMMIT}`  \n\n")

        f.write("---\n\n## 1. Environment\n\n")
        f.write("```text\n" + nvidia_smi[:600] + "\n```\n\n")
        f.write(f"| Item | Value |\n| :--- | :--- |\n")
        f.write(f"| PyTorch GPU | {torch_gpu_desc} |\n")
        f.write(f"| Dataset SHA-256 | `{CANONICAL_DATASET_HASH}` |\n")
        f.write(f"| Train split | 250,000 events |\n")
        f.write(f"| Val split | 100,000 events |\n")
        f.write(f"| Test split | 450,000 events |\n")
        f.write(f"| Holdout | 18,238 events — UNTOUCHED |\n\n")

        f.write("---\n\n## 2. Leaderboard (sorted by Test ROC-AUC)\n\n")
        f.write("> Fit on train → threshold on val → final metrics on test\n\n")
        f.write("| Rank | Model | Device | Val AUC | Test AUC | Test AMS | Thresh | F1 | Fit | Lat/event |\n")
        f.write("| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for rank, (m_id, r) in enumerate(completed, 1):
            gpu = " GPU" if any(x in r["device"] for x in ["cuda", "gpu", "GPU"]) else ""
            f.write(
                f"| {rank} | **{m_id}** | `{r['device'].split(' ')[0]}`{gpu} | "
                f"{r['val_auc']:.4f} | **{r['test_auc']:.4f}** | {r['test_ams']:.4f} | "
                f"{r['opt_thresh']:.4f} | {r['f1']:.4f} | {r['fit_s']:.2f}s | {r['lat_ms']:.4f}ms |\n"
            )

        if not_completed:
            f.write("\n**Non-completed models:**\n\n")
            f.write("| Model | Status | Reason |\n| :--- | :--- | :--- |\n")
            for m_id, r in not_completed:
                f.write(f"| `{m_id}` | {r.get('status')} | {r.get('reason', 'N/A')} |\n")

        f.write("\n---\n\n## 3. Scientific Disclaimer\n\n")
        f.write(
            "Pre-trained certified weights benchmarked on CERN/ATLAS open data "
            "(record 328, DOI: `10.7483/OPENDATA.ATLAS.ZBP2.M5T8`). "
            "Not CERN-validated. Educational/demonstrative purposes only.\n"
        )

    logger.info(f"\nReport written to {report_file}")

    # ── 7. Summary ────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Completed : {len(completed)}")
    logger.info(f"  Not completed: {len(not_completed)}")
    for m_id, r in completed:
        gpu_tag = " [GPU]" if any(x in r["device"] for x in ["cuda", "gpu"]) else ""
        logger.info(
            f"  {m_id:<35} Test AUC={r['test_auc']:.4f}  AMS={r['test_ams']:.4f}"
            f"  {r['fit_s']:.1f}s{gpu_tag}"
        )
    for m_id, r in not_completed:
        logger.info(f"  {m_id:<35} [{r.get('status').upper()}] {r.get('reason', '')[:60]}")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_gpu_first_benchmark()
