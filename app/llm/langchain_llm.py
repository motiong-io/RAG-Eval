import typing as t

from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from ragas.integrations.helicone import helicone_config
from ragas.llms.base import BaseRagasLLM, LangchainLLMWrapper
from ragas.run_config import RunConfig, add_async_retry, add_retry


def llm_factory_w_openai_api(
    model: str = "gpt-4o-mini",
    run_config: t.Optional[RunConfig] = None,
    default_headers: t.Optional[t.Dict[str, str]] = None,
    base_url: t.Optional[str] = None,
    api_key: t.Optional[str] = None,
) -> BaseRagasLLM:
    timeout = None
    if run_config is not None:
        timeout = run_config.timeout

    # if helicone is enabled, use the helicone
    if helicone_config.is_enabled:
        default_headers = helicone_config.default_headers()
        base_url = helicone_config.base_url

    openai_model = ChatOpenAI(
        model=model,
        timeout=timeout,
        default_headers=default_headers,
        api_key=api_key,
        base_url=base_url,
    )
    return LangchainLLMWrapper(openai_model, run_config)
