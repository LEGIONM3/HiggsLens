"""
FastAPI Router for HiggsLens Research Reports & Reproducibility Manifests.
GET /api/v1/events/{event_id}/report
GET /api/v1/reproducibility
"""

import logging
from typing import Literal

from backend.app.schemas.report import EventAnalysisReport, ReproducibilityManifest
from backend.app.services.reporting import reporting_service
from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import HTMLResponse

logger = logging.getLogger("higgslens.api.reports")

router = APIRouter()


@router.get(
    "/events/{event_id}/report",
    response_model=EventAnalysisReport,
    responses={
        200: {
            "description": "Event analysis report in JSON or HTML format",
            "content": {
                "application/json": {},
                "text/html": {},
            },
        },
        404: {"description": "EventId missing or in holdout split"},
        422: {"description": "Unsupported model_id (champion-only xgboost)"},
    },
)
def get_event_report(
    event_id: int,
    format: Literal["json", "html"] = Query(
        "json", description="Export format: 'json' or 'html'"
    ),
    model_id: str = Query(
        "xgboost", description="Champion model ID (strictly 'xgboost')"
    ),
) -> Response:
  """Returns canonical event-analysis report in JSON or print-ready HTML format."""
  if model_id != "xgboost":
    raise HTTPException(
        status_code=422,
        detail=f"Unsupported model '{model_id}'. Reports are champion-only ('xgboost').",
    )

  try:
    report = reporting_service.generate_event_report(
        event_id=event_id, model_id="xgboost"
    )
  except KeyError as e:
    raise HTTPException(status_code=404, detail=str(e))
  except ValueError as e:
    raise HTTPException(status_code=422, detail=str(e))
  except Exception as e:
    logger.error(f"Error generating report for EventId {event_id}: {e}")
    raise HTTPException(
        status_code=500, detail=f"Failed to generate report: {e}"
    )

  if format == "html":
    html_content = reporting_service.render_html_report(report)
    return HTMLResponse(content=html_content)

  return Response(
      content=report.model_dump_json(),
      media_type="application/json",
  )


@router.get("/reproducibility", response_model=ReproducibilityManifest)
def get_reproducibility_manifest() -> ReproducibilityManifest:
  """Returns read-only facts needed to reproduce certified inference."""
  return reporting_service.get_reproducibility_manifest()
