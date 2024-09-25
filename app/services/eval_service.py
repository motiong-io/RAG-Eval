import logging
import os
from typing import List, Optional

import ragas
from datasets import Dataset
from ragas.cost import get_token_usage_for_openai
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.evaluation import Result
from ragas.llms.base import BaseRagasLLM
from ragas.metrics import *

from app.embedding.langchain_embedding import CustomizedLangchainEmbeddingsWrapper
from app.eval.eval_manager import EvalManager
from app.llm.langchain_llm import LangchainLLMFactory
from app.llm.llm_api_price import LLM_API_PRICE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class EvalService:
    def __init__(
        self,
        llm: LangchainLLMFactory,
        embedding: CustomizedLangchainEmbeddingsWrapper,
        eval_manager: EvalManager,
        volume_mount_dir: str,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        reference: Optional[List[str]] = None,
        ground_truths: Optional[List[str]] = None,
    ) -> None:
        self.llm = llm
        self.embedding = embedding
        self.eval_manager = eval_manager
        self.volume_mount_dir = volume_mount_dir
        self.questions = questions if questions is not None else []
        self.ground_truths = ground_truths if ground_truths is not None else []
        self.answers = answers if answers is not None else []
        self.contexts = contexts if contexts is not None else []
        self.reference = reference if reference is not None else []

        # To dict
        self.data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,  # 为解答 question 而从外部知识源检索到的相关上下文。list[list[str]]
            "reference": reference,  # context_precision 和 context_recall 这两个指标的计算需要 reference。这个信息仅在评估 context_precision 和 context_recall 这两个指标时才必须
            "ground_truths": ground_truths,  # question 的标准答案，这是唯一需要人工标注的信息。这个信息仅在评估 context_recall 这一指标时才必须。list[list[str]]
        }

        # Convert dict to dataset
        self.dataset = Dataset.from_dict(self.data)

    def ragas_evaluate(
        self,
        dataset=None,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
    ) -> Result:
        dataset = dataset if dataset is not None else self.dataset

        result = ragas.evaluate(
            llm=self.llm,
            embeddings=self.embedding,
            dataset=dataset,
            metrics=metrics,
            token_usage_parser=get_token_usage_for_openai,
        )

        return result

    def save_result(self, result: Result, output_dir: str, output_file: str):
        """
        Save the result to a specified file in the given directory.

        This function supports saving the result as either a CSV or Excel file,
        depending on the file extension of the output_file parameter.

        Parameters:
        - result (Result): The result object to be saved.
        - output_dir (str): The directory where the output file will be saved.
        - output_file (str): The name of the output file. The file extension should be
                            either '.csv' for CSV files or '.xlsx' or '.xls' for Excel files.

        Raises:
        - ValueError: If the file extension is not supported.
        """
        df = result.to_pandas()
        logger.info(df)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)

        try:
            if output_path.endswith(".csv"):
                df.to_csv(output_path, index=False)
            elif output_path.endswith(".xlsx") or output_path.endswith(".xls"):
                df.to_excel(output_path, index=False)
            else:
                raise ValueError(
                    "Unsupported file extension. Please use .csv, .xlsx, or .xls"
                )

            logger.info(f"Result saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving result: {e}")

    def estimate_token_usage(self, dataset: Dataset = None) -> int:
        dataset = dataset if dataset is not None else self.dataset

        return get_token_usage_for_openai(dataset)

    def run(self) -> None:
        result = self.ragas_evaluate(
            metrics=self.eval_manager.eval_metrics,
        )

        # Customize your output file name
        output_dir = os.path.join(self.volume_mount_dir, "outputs")
        output_file_name = f"ragas_{self.eval_manager.exp_name}_{self.llm.model}_{self.embedding.model}.{self.eval_manager.eval_result_file_ext}"
        self.save_result(result, output_dir, output_file_name)

        # Cost
        try:
            total_tokens = result.total_tokens()
            logger.info(f"Total tokens: {total_tokens}")

            cost_per_input_token, cost_per_output_token = LLM_API_PRICE.get(
                self.llm.model
            )
        except Exception as e:
            logger.error(e)
            logger.info("Token & cost calculation failed.")

        try:
            total_cost = result.total_cost(cost_per_input_token, cost_per_output_token)
            logger.info(f"Total cost: {total_cost}")
        except ValueError as e:
            logger.error(e)
            logger.info(
                "Cost calculation failed. Either due a free local model or  price not rigistered in the llm_api_price.py."
            )
