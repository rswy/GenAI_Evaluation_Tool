# src/metrics/placeholders.py
from .base_metric import BaseMetric
import warnings
# import re # Not used
# import numpy as np # Not used
# import pandas as pd # Not used

class NLIScoreMetric(BaseMetric):
    """Placeholder for NLI-based fact checking. Returns NaN."""
    def compute(self, references, predictions, **kwargs):
        warnings.warn("NLIScoreMetric is a placeholder. Called for single instance.", RuntimeWarning)
        return {"nli_entailment_score": float('nan')}

class LLMAsJudgeFactualityMetric(BaseMetric):
    """Placeholder for using an LLM to judge factuality. Returns NaN."""
    def compute(self, references, predictions, **kwargs):
        warnings.warn("LLMAsJudgeFactualityMetric is a placeholder. Called for single instance.", RuntimeWarning)
        return {"llm_judge_factuality": float('nan')}

class ProfessionalToneMetric(BaseMetric):
    """Placeholder for evaluating professional tone. Returns NaN."""
    def compute(self, references, predictions, **kwargs):
        warnings.warn("ProfessionalToneMetric is a placeholder. Called for single instance.", RuntimeWarning)
        return {"professional_tone_score": float('nan')}

class RefusalQualityMetric(BaseMetric):
    """Placeholder for evaluating refusal appropriateness. Returns NaN."""
    def compute(self, references, predictions, **kwargs):
        warnings.warn("RefusalQualityMetric is a placeholder. Called for single instance.", RuntimeWarning)
        return {"refusal_quality_score": float('nan')}

# Semantic Similarity (COMMENTED OUT) - If re-enabled, it would need similar per-instance refactoring.
# class SemanticSimilarityMetric(BaseMetric):
#     def __init__(self, model_name=DEFAULT_ST_MODEL):
#         # ... init model ...
#         pass
#     def compute(self, references, predictions, **kwargs):
#         # ref_str = str(references)
#         # pred_str = str(predictions)
#         # if not SENTENCE_TRANSFORMERS_AVAILABLE or self.model is None:
#         #     return {"semantic_similarity": float('nan')}
#         # if not ref_str or not pred_str:
#         #     return {"semantic_similarity": 0.0}
#         # try:
#         #      emb_ref = self.model.encode(ref_str, convert_to_tensor=True)
#         #      emb_pred = self.model.encode(pred_str, convert_to_tensor=True)
#         #      cos_sim = util.pytorch_cos_sim(emb_ref, emb_pred)
#         #      return {"semantic_similarity": cos_sim.item()}
#         # except Exception as e:
#         #      warnings.warn(f"SemanticSimilarityMetric: Error for instance: {e}. Assigning 0.", RuntimeWarning)
#         #      return {"semantic_similarity": 0.0}
#         return {"semantic_similarity": float('nan')} # Default if commented out