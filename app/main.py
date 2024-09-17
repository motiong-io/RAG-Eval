# app/main.py
# to use llm
import typing as t

import openai
import ragas
from datasets import Dataset
from ragas import evaluate
from ragas.integrations.helicone import helicone_config
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig, add_async_retry, add_retry

from app.embedding.langchain_embedding import embedding_factory_w_openai_api
from app.llm.langchain_llm import llm_factory_w_openai_api
from app.utils.config import env


def main():
    llm = llm_factory_w_openai_api(
        model="gpt-4o-mini",
        base_url=env.OPENAI_BASE_URL,
        api_key=env.OPENAI_API_KEY,
        default_headers={"Authorization": f"Bearer {env.OPENAI_API_KEY}"},
    )

    embb = embedding_factory_w_openai_api(
        base_url=env.OPENAI_BASE_URL, api_key=env.OPENAI_API_KEY
    )

    questions = [
        "What did the president say about Justice Breyer?",
        "What did the president say about Intel's CEO?",
        "What did the president say about gun violence?",
    ]
    ground_truths = [
        "The president said that Justice Breyer has dedicated his life to serve the country and thanked him for his service.",
        "The president said that Pat Gelsinger is ready to increase Intel's investment to $100 billion.",
        "The president asked Congress to pass proven measures to reduce gun violence.",
    ]
    answers = [
        "What did the president say about Justice Breyer?",
        "What did the president say about Intel's CEO?",
        "What did the president say about gun violence?",
    ]
    contexts = [
        [
            "The president said that Justice Breyer has dedicated his life to serve the country and thanked him for his service."
        ],
        [
            "The president said that Pat Gelsinger is ready to increase Intel's investment to $100 billion."
        ],
        [
            "The president asked Congress to pass proven measures to reduce gun violence."
        ],
    ]
    reference = [
        "The president said that Justice Breyer has dedicated his life to serve the country and thanked him for his service.",
        "The president said that Pat Gelsinger is ready to increase Intel's investment to $100 billion.",
        "The president asked Congress to pass proven measures to reduce gun violence.",
    ]
    # Inference
    for query in questions:
        # answers.append(rag_chain.invoke(query))
        # contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])
        pass

    # To dict
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,  # 为解答 question 而从外部知识源检索到的相关上下文。
        "reference": reference,  # context_precision 和 context_recall 这两个指标的计算需要 reference。这个信息仅在评估 context_precision 和 context_recall 这两个指标时才必须
        "ground_truths": ground_truths,  # question 的标准答案，这是唯一需要人工标注的信息。这个信息仅在评估 context_recall 这一指标时才必须
    }

    # Convert dict to dataset
    dataset = Dataset.from_dict(data)
    result = evaluate(
        llm=llm,
        embeddings=embb,
        dataset=dataset,
        metrics=[
            context_precision,  # 上下文精准度 衡量检索出的上下文中有用信息与无用信息的比率。该指标通过分析 question 和 contexts 来计算。
            context_recall,  # 上下文召回率 用来评估是否检索到了解答问题所需的全部相关信息。这一指标依据 ground_truth（此为框架中唯一基于人工标注的真实数据的指标）和 contexts 进行计算。
            faithfulness,  # 真实性 用于衡量生成答案的事实准确度。它通过对比给定上下文中正确的陈述与生成答案中总陈述的数量来计算。这一指标结合了 question、contexts 和 answer。
            answer_relevancy,  # 答案相关度 评估生成答案与问题的关联程度。例如，对于问题“法国在哪里及其首都是什么？”，答案“法国位于西欧。”的答案相关度较低，因为它只回答了问题的一部分。
        ],
    )

    df = result.to_pandas()
    df.to_csv("output.csv", index=False)  # 保存为CSV文件
    # df.to_excel('output.xlsx', index=False)  # 保存为XSL文件

    print(df)


if __name__ == "__main__":
    main()
