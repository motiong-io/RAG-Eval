import typing as t

from langchain_openai.embeddings import OpenAIEmbeddings
from ragas.embeddings.base import BaseRagasEmbeddings, LangchainEmbeddingsWrapper
from ragas.integrations.helicone import helicone_config
from ragas.run_config import RunConfig, add_async_retry, add_retry


def embedding_factory_w_openai_api(
    model: str = "text-embedding-ada-002",
    run_config: t.Optional[RunConfig] = None,
    base_url: t.Optional[str] = None,
    api_key: t.Optional[str] = None,
) -> BaseRagasEmbeddings:
    openai_embeddings = OpenAIEmbeddings(
        model=model, api_key=api_key, base_url=base_url
    )
    if run_config is not None:
        openai_embeddings.request_timeout = run_config.timeout
    else:
        run_config = RunConfig()
    return LangchainEmbeddingsWrapper(openai_embeddings, run_config=run_config)
