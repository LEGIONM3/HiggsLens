from backend.app.api.v1.endpoints import config, dataset, experiments, health, models, predictions
from backend.app.core.config import settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Research-scale educational and machine-learning platform exploring simulated ATLAS Higgs-to-tau-tau collision events (CERN Open Data Record 328).",
    openapi_url="/api/v1/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount endpoints
app.include_router(health.router, tags=["Health"])
app.include_router(config.router, prefix="/api/v1", tags=["Config"])
app.include_router(dataset.router, prefix="/api/v1/dataset", tags=["Dataset"])
app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])
app.include_router(experiments.router, prefix="/api/v1/experiments", tags=["Experiments"])
app.include_router(predictions.router, prefix="/api/v1/predict", tags=["Predictions"])
