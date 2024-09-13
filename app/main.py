# app/main.py

import ragas
from datasets import Dataset
from app.utils.config import env
import openai
from ragas.llms.base import BaseRagasLLM, llm_factory
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)


def main():
    llm = llm_factory(
        model = "gpt-4o-mini",
        base_url = env.OPENAI_BASE_URL,
        default_headers = { "Authorization": f"Bearer {env.OPENAI_API_KEY}" }
    )
        
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
        dataset = dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
    )

    df = result.to_pandas()


    print(df)

if __name__ == "__main__":
    main()