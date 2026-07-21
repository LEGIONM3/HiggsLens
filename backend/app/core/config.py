import os
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "HiggsLens"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATA_DIR: Path = Path(os.getenv("HIGGSLENS_DATA_DIR", str(BASE_DIR / "data")))
    ARTIFACTS_DIR: Path = Path(os.getenv("HIGGSLENS_ARTIFACTS_DIR", str(BASE_DIR / "artifacts")))
    CONFIG_DIR: Path = Path(os.getenv("HIGGSLENS_CONFIG_DIR", str(BASE_DIR / "configs")))

    # CERN Dataset
    CERN_RECORD_ID: int = 328
    CERN_METADATA_URL: str = "https://opendata.cern.ch/api/records/328"
    CANONICAL_DATASET_URL: str = "https://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz"
    HIGGSLENS_DATASET_URL: Optional[str] = os.getenv("HIGGSLENS_DATASET_URL", None)
    DATASET_FILENAME: str = "atlas-higgs-challenge-2014-v2.csv.gz"

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000", "*"]

    model_config = SettingsConfigDict(env_prefix="HIGGSLENS_", env_file=".env", extra="ignore")

    def get_raw_dataset_path(self) -> Path:
        return self.DATA_DIR / "raw" / self.DATASET_FILENAME


settings = Settings()
