#!/usr/bin/env python
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.experiments.job_manager import job_manager
from backend.app.schemas.experiments import TrainingRequest


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate HiggsLens models reproducibly.")
    parser.add_argument("--mode", type=str, choices=["fast", "research"], default="fast", help="Execution mode (fast=bounded sample/1 seed, research=full dataset/multi seed)")
    parser.add_argument("--models", type=str, help="Comma-separated list of model IDs to train (default: all available)")
    parser.add_argument("--feature-set", type=str, choices=["all_physics", "primary_only", "derived_only"], default="all_physics", help="Feature group to use")
    args = parser.parse_args()

    models_list = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None

    req = TrainingRequest(
        mode=args.mode,
        feature_set=args.feature_set,
        models=models_list
    )

    print(f"Triggering training job in '{args.mode}' mode...")
    job = job_manager.create_job(req)
    print(f"Job created: {job.job_id}")

    # Synchronously poll in CLI until complete
    while True:
        status = job_manager.get_job_status(job.job_id)
        if not status:
            print("Error: Job lost.")
            sys.exit(1)

        print(f"[{status.state.upper()}] {status.progress_message}")
        if status.state == "completed":
            print("\n--- Training Job Completed Successfully ---")
            result = job_manager.get_job_result(job.job_id)
            if result and result.recommendation:
                print(f"Recommended Model: {result.recommendation.recommended_model_id}")
                print(f"Reasoning: {result.recommendation.reasoning}")
            break
        elif status.state == "failed":
            print("\n--- Training Job Failed ---")
            print(status.error_details)
            sys.exit(1)

        time.sleep(2)


if __name__ == "__main__":
    main()
