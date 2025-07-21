from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str
    azure_openai_model_chat: str
    azure_openai_model_embedding: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
