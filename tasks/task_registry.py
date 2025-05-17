# src/tasks/task_registry.py

"""
Defines the supported evaluation tasks and maps them to relevant metrics.
Assumes a flat data structure where each row is one evaluation instance.
"""

# Define task types as constants
RAG_FAQ = "rag_faq"
SUMMARIZATION = "summarization"
CLASSIFICATION = "classification"
CHATBOT = "chatbot"

# --- Define metric names as constants ---
# Fluency & Lexical Similarity
BLEU = "bleu"
ROUGE_1 = "rouge_1"
ROUGE_2 = "rouge_2"
ROUGE_L = "rouge_l"
METEOR = "meteor"

# Semantic Similarity (New)
SEMANTIC_SIMILARITY_SCORE = "semantic_similarity_score"

# Classification
ACCURACY = "accuracy"
PRECISION = "precision"
RECALL = "recall"
F1_SCORE = "f1_score"

# Trust & Factuality
FACT_PRESENCE_SCORE = "fact_presence_score"
# Placeholder for NLI-based fact-checking
NLI_ENTAILMENT_SCORE = "nli_entailment_score" # Placeholder metric (if not already defined)
# Placeholder for LLM-as-judge factuality
LLM_JUDGE_FACTUALITY = "llm_judge_factuality" # Placeholder metric

# Completeness
COMPLETENESS_SCORE = "completeness_score"

# Conciseness
LENGTH_RATIO = "length_ratio"

# Safety & Privacy
SAFETY_KEYWORD_SCORE = "safety_keyword_score"
PII_DETECTION_SCORE = "pii_detection_score"

# Tone & Refusal (Placeholders)
PROFESSIONAL_TONE_SCORE = "professional_tone_score" # Placeholder metric
REFUSAL_QUALITY_SCORE = "refusal_quality_score"     # Placeholder metric


# --- Task to Metric Mapping ---
TASK_METRIC_MAP = {
    RAG_FAQ: [
        BLEU, ROUGE_1, ROUGE_2, ROUGE_L, METEOR,
        SEMANTIC_SIMILARITY_SCORE, # Added
        FACT_PRESENCE_SCORE, COMPLETENESS_SCORE,
        LENGTH_RATIO, SAFETY_KEYWORD_SCORE,
        PII_DETECTION_SCORE,
        PROFESSIONAL_TONE_SCORE, # Placeholder
        REFUSAL_QUALITY_SCORE,     # Placeholder
        NLI_ENTAILMENT_SCORE,      # Placeholder
        LLM_JUDGE_FACTUALITY       # Placeholder
        ],
    SUMMARIZATION: [
        ROUGE_1, ROUGE_2, ROUGE_L, BLEU, METEOR,
        SEMANTIC_SIMILARITY_SCORE, # Added
        COMPLETENESS_SCORE, # Ensure input data has 'ref_key_points' if this is active
        LENGTH_RATIO, SAFETY_KEYWORD_SCORE
        ],
    CLASSIFICATION: [
        ACCURACY, PRECISION, RECALL, F1_SCORE
        ],
    CHATBOT: [
        BLEU, ROUGE_1, ROUGE_2, ROUGE_L, METEOR,
        SEMANTIC_SIMILARITY_SCORE, # Added
        LENGTH_RATIO, SAFETY_KEYWORD_SCORE,
        PII_DETECTION_SCORE,
        PROFESSIONAL_TONE_SCORE, # Placeholder
        REFUSAL_QUALITY_SCORE      # Placeholder
        ],
}

# --- Define Primary Input/Reference Columns per Task ---
PRIMARY_REFERENCE_COLUMN_MAP = {
    RAG_FAQ: "ground_truth", 
    SUMMARIZATION: "ground_truth", 
    CLASSIFICATION: "ground_truth", 
    CHATBOT: "ground_truth"
}
PRIMARY_PREDICTION_COLUMN_MAP = {
    RAG_FAQ: "answer", 
    SUMMARIZATION: "answer", 
    CLASSIFICATION: "answer", 
    CHATBOT: "answer"
}

# --- Define Kwargs for Custom Metrics ---
# These map the internal kwarg name expected by the metric's compute method
# to the column name in the input data file.
CUSTOM_METRIC_KWARG_MAP = {
    "fact_presence_score": {"facts": "ref_facts"}, 
    "completeness_score": {"key_points": "ref_key_points"}
    # Semantic similarity does not require special kwargs from data columns here,
    # as it operates on the primary reference and prediction.
}

# --- Helper Functions ---
def get_metrics_for_task(task_type): 
    return TASK_METRIC_MAP.get(task_type, [])

def get_primary_reference_col(task_type): 
    return PRIMARY_REFERENCE_COLUMN_MAP.get(task_type)

def get_primary_prediction_col(task_type): 
    return PRIMARY_PREDICTION_COLUMN_MAP.get(task_type)

def get_supported_tasks(): 
    return list(TASK_METRIC_MAP.keys())
