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
from app.llm.langchain_llm import LangchainLLMFactory

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class EvalService:
    def __init__(
        self,
        llm: LangchainLLMFactory,
        embedding: CustomizedLangchainEmbeddingsWrapper,
        volume_mount_dir: str,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        reference: Optional[List[str]] = None,
        ground_truths: Optional[List[str]] = None,
        exp_name: str = "ragas",
        file_ext: str = "csv",
    ) -> None:
        self.llm = llm
        self.embedding = embedding
        self.volume_mount_dir = volume_mount_dir
        self.questions = questions if questions is not None else []
        self.ground_truths = ground_truths if ground_truths is not None else []
        self.answers = answers if answers is not None else []
        self.contexts = contexts if contexts is not None else []
        self.reference = reference if reference is not None else []
        self.exp_name = exp_name
        self.file_ext = file_ext
        # To dict
        self.data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,  # 为解答 question 而从外部知识源检索到的相关上下文。
            "reference": reference,  # context_precision 和 context_recall 这两个指标的计算需要 reference。这个信息仅在评估 context_precision 和 context_recall 这两个指标时才必须
            "ground_truths": ground_truths,  # question 的标准答案，这是唯一需要人工标注的信息。这个信息仅在评估 context_recall 这一指标时才必须
        }

        # Convert dict to dataset
        self.dataset = Dataset.from_dict(self.data)

    def evaluate(
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
        result = self.evaluate(
            metrics=[
                context_precision,  # 上下文精准度 衡量检索出的上下文中有用信息与无用信息的比率。该指标通过分析 question 和 contexts 来计算。
                context_recall,  # 上下文召回率 用来评估是否检索到了解答问题所需的全部相关信息。这一指标依据 ground_truth（此为框架中唯一基于人工标注的真实数据的指标）和 contexts 进行计算。
                faithfulness,  # 真实性 用于衡量生成答案的事实准确度。它通过对比给定上下文中正确的陈述与生成答案中总陈述的数量来计算。这一指标结合了 question、contexts 和 answer。
                answer_relevancy,  # 答案相关度 评估生成答案与问题的关联程度。例如，对于问题“法国在哪里及其首都是什么？”，答案“法国位于西欧。”的答案相关度较低，因为它只回答了问题的一部分。
            ],
        )
        # Customize your output file name
        output_dir = os.path.join(self.volume_mount_dir, "outputs")
        output_file_name = f"ragas_{self.exp_name}_{self.llm.model}_{self.embedding.model}.{self.file_ext}"
        self.save_result(result, output_dir, output_file_name)

        # Cost
        print("Total tokens: ", result.total_tokens())
        # print("Total cost: ", result.total_cost())
