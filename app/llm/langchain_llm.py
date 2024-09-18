import typing as t

from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from ragas.integrations.helicone import helicone_config
from ragas.llms.base import BaseRagasLLM, LangchainLLMWrapper
from ragas.run_config import RunConfig, add_async_retry, add_retry


class LangchainLLMFactory(LangchainLLMWrapper):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        run_config: t.Optional[RunConfig] = None,
        base_url: t.Optional[str] = None,
        api_key: t.Optional[str] = None,
        default_headers: t.Optional[t.Dict[str, any]] = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.default_headers = default_headers
        timeout = None
        if run_config is not None:
            timeout = run_config.timeout

        # if helicone is enabled, use the helicone
        if helicone_config.is_enabled:
            self.default_headers = helicone_config.default_headers()
            self.base_url = helicone_config.base_url

        openai_model = ChatOpenAI(
            model=self.model,
            timeout=timeout,
            default_headers=self.default_headers,
            api_key=self.api_key,
            base_url=self.base_url,
        )
        super().__init__(openai_model, run_config)
