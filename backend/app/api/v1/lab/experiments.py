"""
Lab Experiments Management & Execution API Router.
"""

from backend.app.schemas.lab import (
    LabExperimentCreateRequest,
    LabExperimentDetailResponse,
    LabExperimentListResponse,
    LabExperimentSummarySchema,
)
from backend.app.services.lab.job_runner import lab_job_runner
from fastapi import APIRouter, BackgroundTasks

router = APIRouter(prefix="/lab/experiments", tags=["Lab Experiments"])


@router.post(
    "",
    response_model=LabExperimentSummarySchema,
    status_code=202,
    summary="Submit custom training experiment job in sandboxed Lab zone"
)
def create_lab_experiment(
    req: LabExperimentCreateRequest,
    background_tasks: BackgroundTasks
):
    """
    Submits a custom candidate model training experiment in the sandboxed Lab zone.
    Executes model training, threshold selection, and evaluation asynchronously.
    Enforces maximum 1 concurrent job cap (returns 409 Conflict if busy).
    Persists trained artifacts strictly under models/lab_artifacts/{experiment_id}/.
    """
    experiment_id = lab_job_runner.create_and_enqueue_job(
        dataset_id=req.dataset_id,
        model_ids=req.model_ids,
        split_config=req.split_config,
        seed=req.seed,
        sentinel_strategy=req.sentinel_strategy,
    )

    # Launch background job execution
    background_tasks.add_task(lab_job_runner.execute_experiment_job, experiment_id)

    return lab_job_runner.get_experiment_summary(experiment_id)


@router.get(
    "",
    response_model=LabExperimentListResponse,
    summary="List custom lab experiment training jobs"
)
def list_lab_experiments():
    """Lists all recorded custom lab experiments and their status."""
    experiments = lab_job_runner.list_experiments()
    return LabExperimentListResponse(experiments=experiments)


@router.get(
    "/{experiment_id}",
    response_model=LabExperimentDetailResponse,
    summary="Retrieve lab experiment job status and leaderboard metrics"
)
def get_lab_experiment_detail(experiment_id: str):
    """
    Retrieves status, dataset manifest, and per-model EvaluationResult metrics
    for requested lab experiment training job.
    """
    return lab_job_runner.get_experiment_detail(experiment_id)
