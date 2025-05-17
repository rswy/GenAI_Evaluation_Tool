# src/metrics/__init__.py
# This file makes it easier to import metrics and potentially register them

# Import from new dimension-specific files
from .fluency_similarity import BleuMetric, RougeMetric, MeteorMetric, SemanticSimilarityMetric # SemanticSimilarityMetric moved here
from .classification import ClassificationMetrics
from .trust_factuality import FactPresenceMetric
from .completeness import ChecklistCompletenessMetric
from .conciseness import LengthRatioMetric
from .safety import SafetyKeywordMetric, PIIDetectionMetric
from .placeholders import NLIScoreMetric, LLMAsJudgeFactualityMetric, ProfessionalToneMetric, RefusalQualityMetric
# Removed: from .semantic_similarity import SemanticSimilarityMetric 


# Dictionary to map metric names to their respective classes for easy instantiation
# These keys are used internally by get_metric_instances
METRIC_CLASS_REGISTRY = {
    "bleu": BleuMetric, 
    "rouge": RougeMetric, 
    "meteor": MeteorMetric,
    "semantic_similarity": SemanticSimilarityMetric, # Now sourced from fluency_similarity
    "classification": ClassificationMetrics,
    "fact_presence": FactPresenceMetric, 
    "completeness": ChecklistCompletenessMetric,
    "length_ratio": LengthRatioMetric, 
    "safety_keyword": SafetyKeywordMetric,
    "pii_detection": PIIDetectionMetric,
    # --- Placeholders ---
    "professional_tone": ProfessionalToneMetric, 
    "refusal_quality": RefusalQualityMetric,     
    "nli_score": NLIScoreMetric,                 
    "llm_judge_fact": LLMAsJudgeFactualityMetric 
}

# Helper function to get all metric instances needed for a list of metric names
# Metric names here correspond to the *output keys* of the compute methods (e.g., "rouge_1", "accuracy")
def get_metric_instances(metric_names):
    instances = {}
    requested_metric_keys = set(metric_names) # Use a set for faster lookups

    # Standard Metrics
    if "bleu" in requested_metric_keys:
        instances["bleu"] = METRIC_CLASS_REGISTRY["bleu"]()
    if requested_metric_keys.intersection(["rouge_1", "rouge_2", "rouge_l"]):
        instances["rouge"] = METRIC_CLASS_REGISTRY["rouge"]()
    if "meteor" in requested_metric_keys:
        instances["meteor"] = METRIC_CLASS_REGISTRY["meteor"]()
    
    # Semantic Similarity (now grouped with fluency/similarity metrics in terms of file location)
    if "semantic_similarity_score" in requested_metric_keys:
         instances["semantic_similarity"] = METRIC_CLASS_REGISTRY["semantic_similarity"]()
         
    if requested_metric_keys.intersection(["accuracy", "precision", "recall", "f1_score"]):
        instances["classification"] = METRIC_CLASS_REGISTRY["classification"]()

    # Custom/Implemented Metrics
    if "fact_presence_score" in requested_metric_keys:
         instances["fact_presence"] = METRIC_CLASS_REGISTRY["fact_presence"]()
    if "completeness_score" in requested_metric_keys:
         instances["completeness"] = METRIC_CLASS_REGISTRY["completeness"]()
    if "length_ratio" in requested_metric_keys:
         instances["length_ratio"] = METRIC_CLASS_REGISTRY["length_ratio"]()
    if "safety_keyword_score" in requested_metric_keys:
         instances["safety_keyword"] = METRIC_CLASS_REGISTRY["safety_keyword"]()
    if "pii_detection_score" in requested_metric_keys:
        instances["pii_detection"] = METRIC_CLASS_REGISTRY["pii_detection"]()
    
    # --- Placeholders ---
    if "professional_tone_score" in requested_metric_keys:
        instances["professional_tone"] = METRIC_CLASS_REGISTRY["professional_tone"]()
    if "refusal_quality_score" in requested_metric_keys:
        instances["refusal_quality"] = METRIC_CLASS_REGISTRY["refusal_quality"]()
    if "nli_entailment_score" in requested_metric_keys: 
        instances["nli_score"] = METRIC_CLASS_REGISTRY["nli_score"]() 
    if "llm_judge_factuality" in requested_metric_keys: 
         instances["llm_judge_fact"] = METRIC_CLASS_REGISTRY["llm_judge_fact"]()

    return instances
