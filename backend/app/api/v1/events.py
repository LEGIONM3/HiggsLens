"""
FastAPI Router for Event Sampling API (/api/v1/events).
Exposes endpoints to sample and inspect test-split ATLAS events.
"""

import logging
from typing import Optional

from backend.app.schemas.derive import DeriveRequest, DeriveResponse
from backend.app.schemas.events import EventDataResponse, EventSampleResponse
from backend.app.schemas.explain import ExplainResponse
from backend.app.services.derivation import derivation_service
from backend.app.services.event_sampling import (
    EventDatasetNotFoundError,
    event_sampling_service,
)
from backend.app.services.explanation import explanation_service
from fastapi import APIRouter, HTTPException, Query, status

logger = logging.getLogger("higgslens.api.events")

router = APIRouter()


@router.get(
    "/sample",
    response_model=EventSampleResponse,
    status_code=status.HTTP_200_OK,
    summary="Sample test-split ATLAS collision events",
    description=(
        "Deterministically samples n collision events from the official ATLAS test split (KaggleSet=='v'). "
        "Enforces strict holdout isolation and attaches certified champion model prediction probabilities."
    )
)
def sample_events(
    n: int = Query(12, ge=1, le=50, description="Number of events to sample (1 to 50)"),
    seed: int = Query(42, description="Random seed for reproducible sampling"),
    label: str = Query("any", description="Label filter: 'any', 'signal', or 'background'")
) -> EventSampleResponse:
    # Hard cap validation
    if n > 50 or n <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Sample size parameter 'n' must be between 1 and 50."
        )

    label_clean = (label or "any").strip().lower()
    if label_clean not in ("any", "signal", "s", "background", "b"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid label filter '{label}'. Must be 'any', 'signal', or 'background'."
        )

    try:
        return event_sampling_service.sample_events(n=n, seed=seed, label=label_clean)
    except EventDatasetNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error sampling events: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error sampling events: {str(e)}"
        )


@router.get(
    "/{event_id}",
    response_model=EventDataResponse,
    status_code=status.HTTP_200_OK,
    summary="Fetch a specific ATLAS collision event by EventId",
    description=(
        "Fetches a single event by EventId from the official test split. "
        "Returns 404 Not Found if the event is not in the test split (including holdout event IDs)."
    )
)
def get_event_by_id(event_id: int) -> EventDataResponse:
    try:
        event = event_sampling_service.get_event_by_id(event_id)
        if event is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"EventId {event_id} not found in test split."
            )
        return event
    except EventDatasetNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching event {event_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching event {event_id}: {str(e)}"
        )


@router.post(
    "/derive",
    response_model=DeriveResponse,
    status_code=status.HTTP_200_OK,
    summary="Re-derive calculated DER_* features from primary PRI_* inputs",
    description=(
        "Validates 17 primary physics features (PRI_*), re-derives all 13 calculated features (DER_*), "
        "applies MMC policy, and routes the assembled 30-feature vector through the certified PredictionService."
    )
)
def derive_event(request: DeriveRequest) -> DeriveResponse:
    try:
        return derivation_service.derive_and_predict(request)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except EventDatasetNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error deriving event features: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deriving event features: {str(e)}"
        )


@router.get(
    "/{event_id}/explain",
    response_model=ExplainResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute TreeSHAP feature attributions for a specific test-split event by EventId",
    description=(
        "Fetches event features from test split by EventId and computes TreeSHAP feature attributions. "
        "Returns 404 Not Found for missing or holdout event IDs."
    )
)
def explain_event_by_id(event_id: int, model_id: str = "xgboost") -> ExplainResponse:
    try:
        return explanation_service.explain_event_by_id(event_id, model_id=model_id)
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except EventDatasetNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error explaining event {event_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error explaining event {event_id}: {str(e)}"
        )
