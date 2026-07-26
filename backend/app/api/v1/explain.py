"""
API Router for Feature Explanation & Attributions (/api/v1/explain).
"""

import logging

from backend.app.schemas.explain import ExplainRequest, ExplainResponse
from backend.app.services.explanation import explanation_service
from backend.app.services.model_registry import ModelNotFoundError
from fastapi import APIRouter, HTTPException, status

logger = logging.getLogger("higgslens.explain_router")

router = APIRouter(prefix="/explain", tags=["explainability"])


@router.post(
    "",
    response_model=ExplainResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute TreeSHAP feature attributions for a 30-feature event vector",
    description=(
        "Computes exact TreeSHAP attributions in log-odds space using native booster.predict(..., pred_contribs=True). "
        "Validates the Additivity Gate (base_value + sum(contribs) == margin) and aggregates contributions "
        "into 6 canonical physics object groups."
    ),
)
def explain_features(request: ExplainRequest) -> ExplainResponse:
  try:
    return explanation_service.explain(request)
  except ModelNotFoundError as e:
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
    )
  except ValueError as e:
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
    )
  except Exception as e:
    logger.error(f"Unexpected error in explain_features: {e}")
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Error computing feature attributions: {str(e)}",
    )
