import os
import torch
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from typing import Optional

# Force reload environment variables
load_dotenv(override=True)

class Settings(BaseSettings):
    BASE_DIR: Path = Path(__file__).parent
    OUTPUT_DIR: Path = BASE_DIR / "output"
    LOG_DIR: Path = BASE_DIR / "logs"
    
    LATENT_DIM: int = 100
    IMG_SIZE: int = 64
    CHANNELS: int = 3
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.0002
    BETA1: float = 0.5
    
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Yahan hum model ko seedhe environment variable se utha rahe hain
    LLM_MODEL: str = os.getenv("LLM_MODEL", "groq/llama-3.3-70b-versatile")
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY")

    class Config:
        case_sensitive = True

cfg = Settings()
