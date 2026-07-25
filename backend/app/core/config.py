from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "HiggsLens"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    ARTIFACTS_DIR: Path = BASE_DIR / "models" / "artifacts"
    CONFIG_DIR: Path = BASE_DIR / "configs"

    # Lab Sandboxed Zone Paths & Resource Caps
    LAB_DATA_DIR: Path = BASE_DIR / "data" / "lab"
    LAB_ARTIFACTS_DIR: Path = BASE_DIR / "models" / "lab_artifacts"
    LAB_MAX_UPLOAD_SIZE_BYTES: int = 200 * 1024 * 1024  # 200 MB
    LAB_MAX_DATASET_ROWS: int = 500000
    LAB_MAX_MODELS_PER_EXPERIMENT: int = 5
    LAB_JOB_TIMEOUT_SECONDS: int = 300
    LAB_MAX_CONCURRENT_EXPERIMENTS: int = 1

    # CERN ATLAS Dataset Facts
    CERN_RECORD_ID: int = 328
    CERN_DOI: str = "10.7483/OPENDATA.ATLAS.ZBP2.M5T8"
    CERN_SOURCE: str = "CERN/ATLAS open data record 328"
    TOTAL_EVENT_COUNT: int = 818238
    DEFAULT_THRESHOLD: float = 0.6862
    DATASET_FILENAME: str = "atlas-higgs-challenge-2014-v2.csv.gz"
    CERN_METADATA_URL: str = "https://opendata.cern.ch/api/records/328"
    CANONICAL_DATASET_URL: str = "https://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz"
    HIGGSLENS_DATASET_URL: Optional[str] = None

    # CORS configuration for Vite dev server (port 3000 / 5173)
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*"
    ]

    model_config = SettingsConfigDict(env_prefix="HIGGSLENS_", env_file=".env", extra="ignore")

    def get_raw_dataset_path(self) -> Path:
        return self.DATA_DIR / "raw" / self.DATASET_FILENAME


settings = Settings()
