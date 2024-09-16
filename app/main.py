# app/main.py

import ragas
from datasets import Dataset
from app.utils.config import env
import openai
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# to use llm
import typing as t
from ragas.llms.base import BaseRagasLLM, LangchainLLMWrapper
from ragas.embeddings.base import BaseRagasEmbeddings, LangchainEmbeddingsWrapper

from ragas.integrations.helicone import helicone_config
from ragas.run_config import RunConfig, add_async_retry, add_retry
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

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
        base_url=base_url
    )
    return LangchainLLMWrapper(openai_model, run_config)


def embedding_factory_w_openai_api(
    model: str = "text-embedding-ada-002", 
    run_config: t.Optional[RunConfig] = None,
    base_url: t.Optional[str] = None,
    api_key: t.Optional[str] = None,
) -> BaseRagasEmbeddings:
    openai_embeddings = OpenAIEmbeddings(model=model, api_key=api_key, base_url=base_url)
    if run_config is not None:
        openai_embeddings.request_timeout = run_config.timeout
    else:
        run_config = RunConfig()
    return LangchainEmbeddingsWrapper(openai_embeddings, run_config=run_config)

def main():
    llm = llm_factory_w_openai_api(
        model = "gpt-4o-mini",
        base_url = env.OPENAI_BASE_URL,
        api_key = env.OPENAI_API_KEY,
        default_headers = { "Authorization": f"Bearer {env.OPENAI_API_KEY}" }
    )

    embb = embedding_factory_w_openai_api(base_url=env.OPENAI_BASE_URL, api_key=env.OPENAI_API_KEY)
        
    questions = ["What did the president say about Justice Breyer?",
                "What did the president say about Intel's CEO?",
                "What did the president say about gun violence?",
                ]
    ground_truths = [["The president said that Justice Breyer has dedicated his life to serve the country and thanked him for his service."],
                    ["The president said that Pat Gelsinger is ready to increase Intel's investment to $100 billion."],
                    ["The president asked Congress to pass proven measures to reduce gun violence."]]
    answers = ["What did the president say about Justice Breyer?",
                "What did the president say about Intel's CEO?",
                "What did the president say about gun violence?",]
    contexts = [["The president said that Justice Breyer has dedicated his life to serve the country and thanked him for his service."],
                    ["The president said that Pat Gelsinger is ready to increase Intel's investment to $100 billion."],
                    ["The president asked Congress to pass proven measures to reduce gun violence."]]
    reference = ["The president said that Justice Breyer has dedicated his life to serve the country and thanked him for his service.",
                    "The president said that Pat Gelsinger is ready to increase Intel's investment to $100 billion.",
                    "The president asked Congress to pass proven measures to reduce gun violence."]

    # Inference
    for query in questions:
        # answers.append(rag_chain.invoke(query))
        # contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])
        pass

    # To dict
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "reference": reference,
        "ground_truths": ground_truths
    }

    # Convert dict to dataset
    dataset = Dataset.from_dict(data)
    result = evaluate(
        llm=llm,
        embeddings=embb,
        dataset = dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
    )

    df = result.to_pandas()
    df.to_csv('output.csv', index=False)  # 保存为CSV文件



    print(df)

if __name__ == "__main__":
    main()