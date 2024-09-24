from enum import Enum
from typing import Any, Dict, List, Optional

from ragas.metrics import *


class SupportedEvalResultFileExt(str, Enum):
    CSV = "csv"
    XLSX = "xlsx"
    XLS = "xls"


class EvalManager:
    def __init__(
        self,
        eval_result_file_ext: SupportedEvalResultFileExt,
        exp_name: str = "ragas",
        eval_metrics_str: List[str] = [
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
        ],
    ) -> None:
        self.eval_result_file_ext = eval_result_file_ext
        self.exp_name = exp_name
        self.eval_metrics = []
        self.match_metrics(eval_metrics_str)

    def match_metrics(self, eval_metrics_str: List[str]):
        for metric_str in eval_metrics_str:
            if metric_str == "context_precision":
                self.eval_metrics.append(context_precision)
            elif metric_str == "context_recall":
                self.eval_metrics.append(context_recall)
            elif metric_str == "faithfulness":
                self.eval_metrics.append(faithfulness)
            elif metric_str == "answer_relevancy":
                self.eval_metrics.append(answer_relevancy)
            elif metric_str == "answer_correctness":
                self.eval_metrics.append(answer_correctness)
            elif metric_str == "answer_similarity":
                self.eval_metrics.append(answer_similarity)

    # TODO:
    # 1. Check through metrics.ALL_METRICS, whose name equal to the eval_metrics_str, then append to self.eval_metrics
    # 2. More adaptive: use metrics.utils.get_available_metrics to get proper metrics, so that user no need to input metrics manually
