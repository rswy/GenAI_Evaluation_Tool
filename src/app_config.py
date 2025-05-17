# src/app_config.py
"""
Centralized configuration for the Streamlit application.
Includes metric definitions, category information, and UI constants.
"""
from collections import defaultdict

# --- Evaluation Task & Metric Constants (from tasks.task_registry) ---
# It's often good to have a single source of truth.
# For now, we'll redefine or assume these are globally understood if not imported directly
# from tasks.task_registry to keep this module focused on app display config.
# However, for a larger project, importing from task_registry would be better.
RAG_FAQ = "rag_faq"
SUMMARIZATION = "summarization"
CLASSIFICATION = "classification"
CHATBOT = "chatbot"
SEMANTIC_SIMILARITY_SCORE = "semantic_similarity_score"


# --- Metric Display Categories ---
CAT_TRUST = "Trust & Factuality"
CAT_COMPLETENESS = "Completeness & Coverage"
CAT_FLUENCY = "Fluency & Lexical Similarity"
CAT_SEMANTIC = "Semantic Understanding"
CAT_CLASSIFICATION = "Classification Accuracy"
CAT_CONCISENESS = "Conciseness"
CAT_SAFETY = "Safety (Basic Checks)"
CAT_PII_SAFETY = "Privacy/Sensitive Data (Basic Checks)"
CAT_TONE = "Tone & Professionalism"
CAT_REFUSAL = "Refusal Appropriateness"

# --- Descriptions for Metric Categories/Dimensions ---
DIMENSION_DESCRIPTIONS = {
    CAT_TRUST: "Metrics assessing the reliability and factual correctness of the LLM's output, such as the presence of specific, expected factual statements. Placeholder metrics (NLI, LLM-Judge) require advanced setup.",
    CAT_COMPLETENESS: "Metrics evaluating if the LLM response comprehensively addresses all necessary aspects, topics, or key points required by the input query or task instructions.",
    CAT_FLUENCY: "Metrics judging the linguistic quality of the LLM's output, including grammatical correctness, coherence, and similarity to human-like language based on word/phrase overlap (lexical similarity).",
    CAT_SEMANTIC: "Metrics assessing the similarity in meaning (semantic content) between the LLM's output and the reference, going beyond surface-level word matches. Requires sentence-transformers library.",
    CAT_CLASSIFICATION: "Metrics for classification tasks. Per-instance scores indicate correctness for that case; aggregated scores provide overall model performance.",
    CAT_CONCISENESS: "Metrics gauging the brevity and focus of the LLM's response.",
    CAT_SAFETY: "Basic keyword checks for potentially harmful content. These are not exhaustive safety measures. Issues are flagged if detected.",
    CAT_PII_SAFETY: "Basic regex checks for common PII patterns. WARNING: Not a comprehensive PII scan. Issues are flagged if detected.",
    CAT_TONE: "Placeholder metrics for assessing tonal qualities. Full implementation requires dedicated models or human evaluation.",
    CAT_REFUSAL: "Placeholder metrics for evaluating refusal appropriateness. Full implementation requires specific test cases, logic, or human evaluation."
}

# --- Detailed Information for Each Metric ---
METRIC_INFO = {
    # Trust & Factuality
    "fact_presence_score": {
        "name": "Fact Presence",
        "category": CAT_TRUST,
        "higher_is_better": True,
        "explanation": "Checks if specific, predefined factual statements (e.g., 'The Eiffel Tower is in Paris') are explicitly mentioned in the model's answer. Input these exact statements in `ref_facts`. Score: 0-1 (fraction of facts found).",
        "tasks": [RAG_FAQ],
        "input_field_form_label": "Reference Facts (Exact statements to find)",
        "input_field_data_key": "ref_facts"
    },
    "nli_entailment_score": {"name": "NLI Entailment Score", "category": CAT_TRUST, "higher_is_better": True, "explanation": "Placeholder for Natural Language Inference based fact-checking. Currently returns NaN.", "tasks": [RAG_FAQ], "status": "placeholder"},
    "llm_judge_factuality": {"name": "LLM Judge Factuality", "category": CAT_TRUST, "higher_is_better": True, "explanation": "Placeholder for using another LLM to judge factuality. Currently returns NaN.", "tasks": [RAG_FAQ], "status": "placeholder"},

    # Completeness & Coverage
    "completeness_score": {
        "name": "Key Point Coverage",
        "category": CAT_COMPLETENESS,
        "higher_is_better": True,
        "explanation": "Assesses if the model's answer covers a predefined list of broader key topics, concepts, or checklist items (e.g., for a summary: 'main arguments', 'conclusion'). Input these in `ref_key_points`. Score: 0-1 (fraction of points covered).",
        "tasks": [RAG_FAQ, SUMMARIZATION],
        "input_field_form_label": "Reference Key Points/Topics (Broader concepts to cover)",
        "input_field_data_key": "ref_key_points"
    },

    # Fluency & Lexical Similarity
    "bleu": {"name": "BLEU", "category": CAT_FLUENCY, "higher_is_better": True, "explanation": "Measures n-gram (phrase) precision overlap, indicating lexical sequence similarity. Score: 0-1.", "tasks": [RAG_FAQ, SUMMARIZATION, CHATBOT]},
    "rouge_1": {"name": "ROUGE-1 (Unigram)", "category": CAT_FLUENCY, "higher_is_better": True, "explanation": "Measures unigram (single word) recall overlap. Score: 0-1.", "tasks": [RAG_FAQ, SUMMARIZATION, CHATBOT]},
    "rouge_2": {"name": "ROUGE-2 (Bigram)", "category": CAT_FLUENCY, "higher_is_better": True, "explanation": "Measures bigram (two-word phrase) recall overlap. Score: 0-1.", "tasks": [RAG_FAQ, SUMMARIZATION, CHATBOT]},
    "rouge_l": {"name": "ROUGE-L (LCS)", "category": CAT_FLUENCY, "higher_is_better": True, "explanation": "Measures longest common subsequence (LCS) overlap, reflecting sentence-level structural similarity. Score: 0-1.", "tasks": [RAG_FAQ, SUMMARIZATION, CHATBOT]},
    "meteor": {"name": "METEOR", "category": CAT_FLUENCY, "higher_is_better": True, "explanation": "Harmonic mean of unigram precision/recall, with stemming/synonym matching. Score: 0-1.", "tasks": [RAG_FAQ, SUMMARIZATION, CHATBOT]},

    # Semantic Understanding
    SEMANTIC_SIMILARITY_SCORE: {"name": "Semantic Similarity", "category": CAT_SEMANTIC, "higher_is_better": True, "explanation": "Measures similarity in meaning using sentence embeddings. Score typically -1 to 1 (often 0 to 1 for positive similarity). Requires 'sentence-transformers' library.", "tasks": [RAG_FAQ, SUMMARIZATION, CHATBOT]},

    # Classification Accuracy
    "accuracy": {"name": "Accuracy (Classification)", "category": CAT_CLASSIFICATION, "higher_is_better": True, "explanation": "Per-instance: 1.0 if correct, 0.0 otherwise. Aggregated: overall percentage correct.", "tasks": [CLASSIFICATION]},
    "precision": {"name": "Precision (Classification)", "category": CAT_CLASSIFICATION, "higher_is_better": True, "explanation": "Per-instance: 1.0/0.0 (simplified view). Aggregated: standard precision (TP / (TP + FP)).", "tasks": [CLASSIFICATION]},
    "recall": {"name": "Recall (Classification)", "category": CAT_CLASSIFICATION, "higher_is_better": True, "explanation": "Per-instance: 1.0/0.0 (simplified view). Aggregated: standard recall (TP / (TP + FN)).", "tasks": [CLASSIFICATION]},
    "f1_score": {"name": "F1-Score (Classification)", "category": CAT_CLASSIFICATION, "higher_is_better": True, "explanation": "Per-instance: 1.0/0.0 (simplified view). Aggregated: standard F1-score.", "tasks": [CLASSIFICATION]},

    # Conciseness
    "length_ratio": {"name": "Length Ratio", "category": CAT_CONCISENESS, "higher_is_better": False, "explanation": "Ratio of model answer's word count to ground truth's. Closer to 1.0 often desired.", "tasks": [RAG_FAQ, SUMMARIZATION, CHATBOT]},

    # Safety (Basic Checks)
    "safety_keyword_score": {"name": "Safety Keyword Score", "category": CAT_SAFETY, "higher_is_better": True, "explanation": "Basic check: 1.0 if NO predefined unsafe keywords found, 0.0 if ANY found. Issues are flagged if score is < 1.0.", "tasks": [RAG_FAQ, SUMMARIZATION, CHATBOT]},

    # Privacy/Sensitive Data (Basic Checks)
    "pii_detection_score": {"name": "PII Detection Score", "category": CAT_PII_SAFETY, "higher_is_better": True, "explanation": "Basic regex check: 1.0 if NO common PII patterns found, 0.0 if ANY found. Issues are flagged if score is < 1.0. Not a comprehensive scan.", "tasks": [RAG_FAQ, CHATBOT]},

    # Tone & Professionalism
    "professional_tone_score": {"name": "Professional Tone", "category": CAT_TONE, "higher_is_better": True, "explanation": "Placeholder for professional tone evaluation. Currently returns NaN.", "tasks": [RAG_FAQ, CHATBOT], "status": "placeholder"},

    # Refusal Appropriateness
    "refusal_quality_score": {"name": "Refusal Quality", "category": CAT_REFUSAL, "higher_is_better": True, "explanation": "Placeholder for evaluating refusal appropriateness. Currently returns NaN.", "tasks": [RAG_FAQ, CHATBOT], "status": "placeholder"},
}

# --- UI Display Order and Grouping ---
METRICS_BY_CATEGORY = defaultdict(list)
CATEGORY_ORDER = [
    CAT_TRUST, CAT_COMPLETENESS, CAT_FLUENCY, CAT_SEMANTIC, CAT_CLASSIFICATION,
    CAT_CONCISENESS, CAT_SAFETY, CAT_PII_SAFETY, CAT_TONE, CAT_REFUSAL
]
for key, info in METRIC_INFO.items():
    METRICS_BY_CATEGORY[info['category']].append(key)

# Ensure all categories in METRIC_INFO are in CATEGORY_ORDER, add any missing ones at the end
for cat_key_info in METRIC_INFO.values():
    cat = cat_key_info['category']
    if cat not in CATEGORY_ORDER:
        CATEGORY_ORDER.append(cat)

# --- Constants for Data Editor Form ---
REQUIRED_FIELDS_ADD_ROW = ['task_type', 'model', 'question', 'ground_truth', 'answer']
OPTIONAL_FIELDS_ADD_ROW_INFO = {
    "ref_facts": {
        "label": "Reference Facts (Exact statements to find, comma-separated)",
        "placeholder": "e.g., The sky is blue,Earth is round",
        "metric_info": "For Fact Presence: Checks for these specific statements in the answer."
    },
    "ref_key_points": {
        "label": "Reference Key Points/Topics (Broader concepts to cover, comma-separated)",
        "placeholder": "e.g., main historical events,key product features,pros and cons",
        "metric_info": "For Key Point Coverage: Checks if these general topics are addressed."
    },
    "test_description": {"label": "Test Description", "placeholder": "Briefly describe this test case's purpose...", "metric_info": "Optional metadata"}
}

# --- Key Metrics for "Best Model Summary" Highlights ---
# This could also be part of app_config
KEY_METRICS_PER_TASK_FOR_HIGHLIGHTS = {
    RAG_FAQ: [SEMANTIC_SIMILARITY_SCORE, "fact_presence_score", "completeness_score", "rouge_l"],
    SUMMARIZATION: [SEMANTIC_SIMILARITY_SCORE, "completeness_score", "rouge_l", "length_ratio"],
    CLASSIFICATION: ["f1_score", "accuracy"],
    CHATBOT: [SEMANTIC_SIMILARITY_SCORE, "meteor", "rouge_l", "length_ratio"]
}
