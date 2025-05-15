# src/tasks/task_registry.py (Flat Format - Semantic Similarity Disabled)

"""
Defines the supported evaluation tasks and maps them to relevant metrics.
Assumes a flat data structure where each row is one evaluation instance.
Semantic Similarity is disabled by default due to potential performance impact / PyTorch errors.
"""

# Define task types as constants
RAG_FAQ = "rag_faq"
SUMMARIZATION = "summarization"
CLASSIFICATION = "classification"
CHATBOT = "chatbot"

# --- Define metric names as constants ---
BLEU = "bleu"; ROUGE_1 = "rouge_1"; ROUGE_2 = "rouge_2"; ROUGE_L = "rouge_l"; METEOR = "meteor"
ACCURACY = "accuracy"; PRECISION = "precision"; RECALL = "recall"; F1_SCORE = "f1_score"
FACT_PRESENCE_SCORE = "fact_presence_score"; COMPLETENESS_SCORE = "completeness_score"
LENGTH_RATIO = "length_ratio"; SAFETY_KEYWORD_SCORE = "safety_keyword_score"
# SEMANTIC_SIMILARITY = "semantic_similarity" # COMMENTED OUT

# --- Define New Constants ---
PII_DETECTION_SCORE = "pii_detection_score"
PROFESSIONAL_TONE_SCORE = "professional_tone_score" # Placeholder metric
REFUSAL_QUALITY_SCORE = "refusal_quality_score"     # Placeholder metric
NLI_ENTAILMENT_SCORE = "nli_entailment_score"       # Placeholder metric (if not already defined)



# --- Task to Metric Mapping (Semantic Similarity commented out) ---
TASK_METRIC_MAP = {

    RAG_FAQ: [
        BLEU, ROUGE_1, ROUGE_2, ROUGE_L, METEOR,
        FACT_PRESENCE_SCORE, COMPLETENESS_SCORE,
        LENGTH_RATIO, SAFETY_KEYWORD_SCORE,
        # SEMANTIC_SIMILARITY, # COMMENTED OUT

        # --- Add New Metrics ---
        PII_DETECTION_SCORE,
        PROFESSIONAL_TONE_SCORE, # Add placeholder if desired
        REFUSAL_QUALITY_SCORE,     # Add placeholder if desired
        NLI_ENTAILMENT_SCORE       # Add placeholder if desired
        ],
    SUMMARIZATION: [
        ROUGE_1, ROUGE_2, ROUGE_L, BLEU, METEOR,
        # SEMANTIC_SIMILARITY, # COMMENTED OUT
        COMPLETENESS_SCORE, # Ensure input data has 'ref_key_points' if this is active
        LENGTH_RATIO, SAFETY_KEYWORD_SCORE
        ],
    CLASSIFICATION: [
        ACCURACY, PRECISION, RECALL, F1_SCORE
        ],
    CHATBOT: [
        BLEU, ROUGE_1, ROUGE_2, ROUGE_L, METEOR,
        # SEMANTIC_SIMILARITY, # COMMENTED OUT
        LENGTH_RATIO, SAFETY_KEYWORD_SCORE
        ],
}

# --- Define Primary Input/Reference Columns per Task ---
PRIMARY_REFERENCE_COLUMN_MAP = {RAG_FAQ: "ground_truth", SUMMARIZATION: "ground_truth", CLASSIFICATION: "ground_truth", CHATBOT: "ground_truth"}
PRIMARY_PREDICTION_COLUMN_MAP = {RAG_FAQ: "answer", SUMMARIZATION: "answer", CLASSIFICATION: "answer", CHATBOT: "answer"}

# --- Define Kwargs for Custom Metrics ---
CUSTOM_METRIC_KWARG_MAP = {"fact_presence_score": {"facts": "ref_facts"}, "completeness_score": {"key_points": "ref_key_points"}}

# --- Helper Functions ---
def get_metrics_for_task(task_type): return TASK_METRIC_MAP.get(task_type, [])
def get_primary_reference_col(task_type): return PRIMARY_REFERENCE_COLUMN_MAP.get(task_type)
def get_primary_prediction_col(task_type): return PRIMARY_PREDICTION_COLUMN_MAP.get(task_type)
def get_supported_tasks(): return list(TASK_METRIC_MAP.keys())