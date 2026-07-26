from backend.app.api.v1 import dataset, health, metrics, models, predict
from backend.app.core.config import settings
from backend.app.core.logging import setup_logging
from backend.app.services.model_registry import ArtifactCorruptError, ModelNotFoundError
from backend.app.services.prediction_service import FeatureValidationError
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


def create_app() -> FastAPI:
    """App-factory for HiggsLens FastAPI backend service layer (inference-only)."""
    setup_logging()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="FastAPI service layer for statistical Higgs-event classification on CERN/ATLAS open data.",
        openapi_url="/api/v1/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Configure CORS for Vite dev server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routers
    app.include_router(health.router)
    app.include_router(dataset.router, prefix="/api/v1/dataset")
    app.include_router(models.router, prefix="/api/v1/models")
    app.include_router(predict.router, prefix="/api/v1/predict")
    app.include_router(metrics.router, prefix="/api/v1/metrics")

    from backend.app.api.v1 import events
    app.include_router(events.router, prefix="/api/v1/events")

    # Lab Sandboxed Zone Routers
    from backend.app.api.v1.lab import datasets_router, experiments_router
    app.include_router(datasets_router, prefix="/api/v1")
    app.include_router(experiments_router, prefix="/api/v1")

    # Error Contract Exception Handlers
    @app.exception_handler(ModelNotFoundError)
    async def model_not_found_handler(request: Request, exc: ModelNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"detail": f"Unknown model_id '{exc.model_id}'."}
        )

    @app.exception_handler(ArtifactCorruptError)
    async def artifact_corrupt_handler(request: Request, exc: ArtifactCorruptError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": f"Artifact missing or corrupt for model '{exc.model_id}': {exc.message}"}
        )

    @app.exception_handler(FeatureValidationError)
    async def feature_validation_handler(request: Request, exc: FeatureValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": exc.message, "details": exc.details}
        )

    @app.exception_handler(RequestValidationError)
    async def pydantic_validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": "Unprocessable Entity: invalid feature schema or payload structure.", "errors": exc.errors()}
        )

    return app


app = create_app()
