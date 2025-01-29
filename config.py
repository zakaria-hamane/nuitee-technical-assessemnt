from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from dotenv import load_dotenv
import os

load_dotenv()


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file="example.env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # API Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    MODEL_NAME: str = "gpt-4o"  # This model supports JSON mode
    API_PORT: int = 8000
    DEBUG_MODE: bool = True

    # Matching Settings
    SIMILARITY_THRESHOLD: float = 0.6
    FEATURE_WEIGHT: float = 0.7
    TEXT_WEIGHT: float = 0.3

    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


settings = Settings()