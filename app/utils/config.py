import os
from typing import Optional

from pydantic import PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSettings(BaseSettings):
    # this is to take in all the environment variables
    # this is for the logging purpose
    # when deployed on cloud, the name and ver should come from envvar

    # some from configmap
    ENV: str
    SERVICE_NAME: str
    SERVICE_VERSION: str

    MOUNT_PATH: str

    # object storage
    OBJECT_STORAGE_ENDPOINT_URL: str
    # if oneday we are on azure, azure only requires conn string
    # no need below 2 items
    OBJECT_STORAGE_ACCESS_KEY: Optional[str] = None
    OBJEST_STORAGE_SECRET_KEY: Optional[str] = None
    BUCKET_NAME: str

    WORKFLOW_SERVICE_DOMAIN: str
    WORKFLOW_SERVICE_PORT: int

    # openai
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str

    # eval
    EVAL_LLM: str = "gpt-4o-mini"
    EVAL_EMBEDDING: str = "text-embedding-3-small"
    FILE_EXT: str = "csv"
    EXP_NAME: str = "ragas"
    EVAL_METRICS: list[str] = [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
    ]


class LocalDevSettings(EnvSettings):
    # it reads from a config file at root
    # this config file is gitignored
    # this config file needs to have a template
    model_config = SettingsConfigDict(env_file="config", extra="ignore")


class DeployedSettings(EnvSettings):
    # takes in env vars from the pod
    ...


def find_config() -> EnvSettings:
    if os.getenv("ENV"):
        return DeployedSettings()
    else:
        return LocalDevSettings()


env = find_config()


if __name__ == "__main__":
    print(env)
