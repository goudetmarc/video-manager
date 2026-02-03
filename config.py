"""Configuration: chemin du dossier vidéo (NAS)."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    video_root: str = ""
    supabase_url: str = ""
    supabase_key: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
