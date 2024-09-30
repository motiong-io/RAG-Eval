# app/main.py
# to use llm
import os
import typing as t

import openai
import ragas
from datasets import Dataset
from ragas.integrations.helicone import helicone_config
from ragas.run_config import RunConfig, add_async_retry, add_retry

from app.embedding.langchain_embedding import CustomizedLangchainEmbeddingsWrapper
from app.eval.eval_manager import EvalManager
from app.llm.langchain_llm import LangchainLLMFactory
from app.services.eval_service import EvalService
from app.utils.config import env
from tests.qa_example5_agentx_correct_ref import answers, contexts, ground_truths, questions, reference


def main():
    # #Your RAG/KG Q & A, etc here
    # questions = [
    #     "What did the president say about Justice Breyer?",
    #     "What did the president say about Intel's CEO?",
    #     "What did the president say about gun violence?",
    # ]

    # #ground_truths can be a list of strings(recommended), or a list of lists
    #     "The president said that Justice Breyer has dedicated his life to serve the country and thanked him for his service.",
    #     "The president said that Pat Gelsinger is ready to increase Intel's investment to $100 billion.",
    #     "The president asked Congress to pass proven measures to reduce gun violence.",
    # ]

    # answers = [
    #     "What did the president say about Justice Breyer?",
    #     "What did the president say about Intel's CEO?",
    #     "What did the president say about gun violence?",
    # ]

    # #contexts are the retrieved chunks
    # contexts = [
    #     [
    #         "The president said that Justice Breyer has dedicated his life to serve the country and thanked him for his service."
    #     ],
    #     [
    #         "The president said that Pat Gelsinger is ready to increase Intel's investment to $100 billion."
    #     ],
    #     [
    #         "The president asked Congress to pass proven measures to reduce gun violence."
    #     ],
    # ]

    # #Required for answer_correctness, answer_similarity, context_precision, context_recall. 
    # # Not required for faithfulness, answer_relevancy.
    # # It could be the true chunks to the true answer
    # reference = [
    #     # "The president said that Justice Breyer has dedicated his life to serve the country and thanked him for his service.",
    #     # "The president said that Pat Gelsinger is ready to increase Intel's investment to $100 billion.",
    #     # "The president asked Congress to pass proven measures to reduce gun violence.",
    # ]
    # Inference
    for query in questions:
        # answers.append(rag_chain.invoke(query))
        # contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])
        pass

    # RAGAs evaluation
    llm = LangchainLLMFactory(
        model=env.EVAL_LLM,
        base_url=env.OPENAI_BASE_URL,
        api_key=env.OPENAI_API_KEY,
    )

    embb = CustomizedLangchainEmbeddingsWrapper(
        model=env.EVAL_EMBEDDING,
        base_url=env.OPENAI_BASE_URL,
        api_key=env.OPENAI_API_KEY,
    )

    eval_manager = EvalManager(
        exp_name=env.EXP_NAME,
        eval_result_file_ext=env.FILE_EXT,
        eval_metrics_str=env.EVAL_METRICS,
    )

    s = EvalService(
        llm=llm,
        embedding=embb,
        eval_manager=eval_manager,
        volume_mount_dir=env.MOUNT_PATH,
        questions=questions,
        answers=answers,
        contexts=contexts,
        reference=reference,
        ground_truths=ground_truths,
    )

    s.run()


if __name__ == "__main__":
    main()
