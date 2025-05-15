# src/metrics/__init__.py
# This file makes it easier to import metrics and potentially register them

# Import from new dimension-specific files
from .fluency_similarity import BleuMetric, RougeMetric, MeteorMetric
from .classification import ClassificationMetrics
from .trust_factuality import FactPresenceMetric # Assuming NLIScoreMetric, LLMJudge moved to placeholders
from .completeness import ChecklistCompletenessMetric
from .conciseness import LengthRatioMetric
from .safety import SafetyKeywordMetric, PIIDetectionMetric
from .placeholders import NLIScoreMetric, LLMAsJudgeFactualityMetric, ProfessionalToneMetric, RefusalQualityMetric # Import placeholders



# Dictionary to map metric names to their respective classes for easy instantiation
# These keys are used internally by get_metric_instances
METRIC_CLASS_REGISTRY = {
    "bleu": BleuMetric, 
    "rouge": RougeMetric, 
    "meteor": MeteorMetric,
    "classification": ClassificationMetrics,
    "fact_presence": FactPresenceMetric, 
    "completeness": ChecklistCompletenessMetric,
    "length_ratio": LengthRatioMetric, 
    "safety_keyword": SafetyKeywordMetric,
    # "semantic_similarity": SemanticSimilarityMetric, # Assumed commented out
    # --- Add New Metrics ---
    "pii_detection": PIIDetectionMetric,
    "professional_tone": ProfessionalToneMetric, # Placeholder
    "refusal_quality": RefusalQualityMetric,     # Placeholder
    # --- Placeholders ---
    "nli_score": NLIScoreMetric,                 # Placeholder (if key used elsewhere)
    "llm_judge_fact": LLMAsJudgeFactualityMetric # Placeholder (if key used elsewhere)
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
    
    # if "semantic_similarity" in requested_metric_keys: # Assumed commented out
    #      instances["semantic_similarity"] = METRIC_CLASS_REGISTRY["semantic_similarity"]()

    # --- Add New Metrics ---
    if "pii_detection_score" in requested_metric_keys:
        instances["pii_detection"] = METRIC_CLASS_REGISTRY["pii_detection"]()
    if "professional_tone_score" in requested_metric_keys:
        instances["professional_tone"] = METRIC_CLASS_REGISTRY["professional_tone"]()
    if "refusal_quality_score" in requested_metric_keys:
        instances["refusal_quality"] = METRIC_CLASS_REGISTRY["refusal_quality"]()

    # --- Handle existing placeholders if needed ---
    if "nli_entailment_score" in requested_metric_keys:
        instances["nli_score"] = NLIScoreMetric() # Instantiate directly if not in registry map
    if "llm_judge_factuality" in requested_metric_keys:
         instances["llm_judge_fact"] = LLMAsJudgeFactualityMetric() # Instantiate directly

    return instances