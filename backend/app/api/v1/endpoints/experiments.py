import json
from typing import List

from backend.app.core.config import settings
from backend.app.experiments.job_manager import job_manager
from backend.app.schemas.experiments import (
    ExperimentRunResponse,
    JobStatusResponse,
    TrainingRequest,
)
from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.post("/train", response_model=JobStatusResponse)
def trigger_training(request: TrainingRequest):
    # Verify dataset exists before starting job
    raw_path = settings.get_raw_dataset_path()
    if not raw_path.exists():
        raise HTTPException(status_code=400, detail="CERN dataset not found. Please download it first via POST /api/v1/dataset/download.")
    return job_manager.create_job(request)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    job = job_manager.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@router.get("/jobs", response_model=List[JobStatusResponse])
def list_jobs():
    return job_manager.list_jobs()


@router.get("", response_model=List[ExperimentRunResponse])
def list_experiments():
    log_file = settings.ARTIFACTS_DIR / "experiments_log.jsonl"
    runs: List[ExperimentRunResponse] = []
    if log_file.exists():
        with open(log_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    runs.append(ExperimentRunResponse.model_validate(data))
                except Exception:
                    continue
    return runs


@router.get("/{run_id}", response_model=ExperimentRunResponse)
def get_experiment_run(run_id: str):
    log_file = settings.ARTIFACTS_DIR / "experiments_log.jsonl"
    if log_file.exists():
        with open(log_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if data.get("run_id") == run_id:
                        return ExperimentRunResponse.model_validate(data)
                except Exception:
                    continue
    raise HTTPException(status_code=404, detail=f"Experiment run '{run_id}' not found.")
