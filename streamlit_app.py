# # streamlit_app.py
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from pathlib import Path
# import sys
# import os
# import datetime
# import tempfile
# import json
# import numpy as np
# from collections import defaultdict
# import warnings
# import matplotlib # Keep for colormap
# import matplotlib.cm as cm # Keep for colormap
# import copy # Keep for deepcopy if used
# import traceback # For error logging

# # --- Add project root to sys.path ---
# project_root = Path(__file__).resolve().parent 
# data_dir = project_root / "data"
# src_path = project_root / "src"
# if str(src_path) not in sys.path:
#     sys.path.insert(0, str(src_path))

# # --- Import framework functions ---
# try:
#     from data_loader import load_data
#     from evaluator import evaluate_model_responses 
#     from file_converter import convert_excel_to_data, convert_csv_to_data
#     from mock_data_generator import generate_mock_data_flat 
#     from tasks.task_registry import (
#         get_metrics_for_task, get_supported_tasks, 
#         RAG_FAQ, SUMMARIZATION, CLASSIFICATION, CHATBOT, 
#         CUSTOM_METRIC_KWARG_MAP, SEMANTIC_SIMILARITY_SCORE 
#     )
# except ImportError as e:
#     st.error(f"Framework Import Error: {e}. Please ensure all necessary files are in the 'src' directory and Python environment is set up correctly.")
#     st.error(f"Current sys.path: {sys.path}")
#     st.error(f"Project root evaluated as: {project_root}")
#     st.stop()

# # --- Metric Information (for display purposes) ---
# CAT_TRUST = "Trust & Factuality"
# CAT_COMPLETENESS = "Completeness & Coverage" # Renamed for clarity
# CAT_FLUENCY = "Fluency & Lexical Similarity" 
# CAT_SEMANTIC = "Semantic Understanding"
# CAT_CLASSIFICATION = "Classification Accuracy"
# CAT_CONCISENESS = "Conciseness"
# CAT_SAFETY = "Safety (Basic Checks): <Not Yet Implemented>"
# CAT_PII_SAFETY = "Privacy/Sensitive Data (Basic Checks) <Not Yet Implemented>"
# CAT_TONE = "Tone & Professionalism <Not Yet Implemented>" 
# CAT_REFUSAL = "Refusal Appropriateness <Not Yet Implemented>" 

# DIMENSION_DESCRIPTIONS = {
#     CAT_TRUST: "Metrics assessing the reliability and factual correctness of the LLM's output, such as the presence of specific, expected factual statements. Placeholder metrics (NLI, LLM-Judge) require advanced setup.",
#     CAT_COMPLETENESS: "Metrics evaluating if the LLM response comprehensively addresses all necessary aspects, topics, or key points required by the input query or task instructions.",
#     CAT_FLUENCY: "Metrics judging the linguistic quality of the LLM's output, including grammatical correctness, coherence, and similarity to human-like language based on word/phrase overlap (lexical similarity).",
#     CAT_SEMANTIC: "Metrics assessing the similarity in meaning (semantic content) between the LLM's output and the reference, going beyond surface-level word matches. Requires sentence-transformers library.",
#     CAT_CLASSIFICATION: "Metrics for classification tasks. Per-instance scores indicate correctness for that case; aggregated scores provide overall model performance.",
#     CAT_CONCISENESS: "Metrics gauging the brevity and focus of the LLM's response.",
#     CAT_SAFETY: "Basic keyword checks for potentially harmful content. These are not exhaustive safety measures. Issues are flagged if detected.",
#     CAT_PII_SAFETY: "Basic regex checks for common PII patterns. WARNING: Not a comprehensive PII scan. Issues are flagged if detected.",
#     CAT_TONE: "Placeholder metrics for assessing tonal qualities. Full implementation requires dedicated models or human evaluation.",
#     CAT_REFUSAL: "Placeholder metrics for evaluating refusal appropriateness. Full implementation requires specific test cases, logic, or human evaluation."
# }

# METRIC_INFO = { 
#     # Trust & Factuality
#     "fact_presence_score": {
#         "name": "Fact Presence", 
#         "category": CAT_TRUST, 
#         "higher_is_better": True, 
#         "explanation": "Checks if specific, predefined factual statements (e.g., 'The Eiffel Tower is in Paris') are explicitly mentioned in the model's answer. Input these exact statements in `ref_facts`. Score: 0-1 (fraction of facts found).", 
#         "tasks": [RAG_FAQ], 
#         "input_field_form_label": "Reference Facts (Exact statements to find)", 
#         "input_field_data_key": "ref_facts"
#     },
#     "nli_entailment_score": {"name": "NLI Entailment Score", "category": CAT_TRUST, "higher_is_better": True, "explanation": "Placeholder for Natural Language Inference based fact-checking. Currently returns NaN.", "tasks": [RAG_FAQ], "status": "placeholder"},
#     "llm_judge_factuality": {"name": "LLM Judge Factuality", "category": CAT_TRUST, "higher_is_better": True, "explanation": "Placeholder for using another LLM to judge factuality. Currently returns NaN.", "tasks": [RAG_FAQ], "status": "placeholder"},
    
#     # Completeness & Coverage
#     "completeness_score": {
#         "name": "Key Point Coverage", # Renamed for clarity
#         "category": CAT_COMPLETENESS, 
#         "higher_is_better": True, 
#         "explanation": "Assesses if the model's answer covers a predefined list of broader key topics, concepts, or checklist items (e.g., for a summary: 'main arguments', 'conclusion'). Input these in `ref_key_points`. Score: 0-1 (fraction of points covered).", 
#         "tasks": [RAG_FAQ, SUMMARIZATION], 
#         "input_field_form_label": "Reference Key Points/Topics (Broader concepts to cover)", 
#         "input_field_data_key": "ref_key_points"
#     },
    
#     # Fluency & Lexical Similarity
#     "bleu": {"name": "BLEU", "category": CAT_FLUENCY, "higher_is_better": True, "explanation": "Measures n-gram (phrase) precision overlap, indicating lexical sequence similarity. Score: 0-1.", "tasks": [RAG_FAQ, SUMMARIZATION, CHATBOT]},
#     "rouge_1": {"name": "ROUGE-1 (Unigram)", "category": CAT_FLUENCY, "higher_is_better": True, "explanation": "Measures unigram (single word) recall overlap. Score: 0-1.", "tasks": [RAG_FAQ, SUMMARIZATION, CHATBOT]},
#     "rouge_2": {"name": "ROUGE-2 (Bigram)", "category": CAT_FLUENCY, "higher_is_better": True, "explanation": "Measures bigram (two-word phrase) recall overlap. Score: 0-1.", "tasks": [RAG_FAQ, SUMMARIZATION, CHATBOT]},
#     "rouge_l": {"name": "ROUGE-L (LCS)", "category": CAT_FLUENCY, "higher_is_better": True, "explanation": "Measures longest common subsequence (LCS) overlap, reflecting sentence-level structural similarity. Score: 0-1.", "tasks": [RAG_FAQ, SUMMARIZATION, CHATBOT]},
#     "meteor": {"name": "METEOR", "category": CAT_FLUENCY, "higher_is_better": True, "explanation": "Harmonic mean of unigram precision/recall, with stemming/synonym matching. Score: 0-1.", "tasks": [RAG_FAQ, SUMMARIZATION, CHATBOT]},
    
#     # Semantic Understanding
#     SEMANTIC_SIMILARITY_SCORE: {"name": "Semantic Similarity", "category": CAT_SEMANTIC, "higher_is_better": True, "explanation": "Measures similarity in meaning using sentence embeddings. Score typically -1 to 1 (often 0 to 1 for positive similarity). Requires 'sentence-transformers' library.", "tasks": [RAG_FAQ, SUMMARIZATION, CHATBOT]},

#     # Classification Accuracy
#     "accuracy": {"name": "Accuracy (Classification)", "category": CAT_CLASSIFICATION, "higher_is_better": True, "explanation": "Per-instance: 1.0 if correct, 0.0 otherwise. Aggregated: overall percentage correct.", "tasks": [CLASSIFICATION]},
#     "precision": {"name": "Precision (Classification)", "category": CAT_CLASSIFICATION, "higher_is_better": True, "explanation": "Per-instance: 1.0/0.0 (simplified view). Aggregated: standard precision (TP / (TP + FP)).", "tasks": [CLASSIFICATION]},
#     "recall": {"name": "Recall (Classification)", "category": CAT_CLASSIFICATION, "higher_is_better": True, "explanation": "Per-instance: 1.0/0.0 (simplified view). Aggregated: standard recall (TP / (TP + FN)).", "tasks": [CLASSIFICATION]},
#     "f1_score": {"name": "F1-Score (Classification)", "category": CAT_CLASSIFICATION, "higher_is_better": True, "explanation": "Per-instance: 1.0/0.0 (simplified view). Aggregated: standard F1-score.", "tasks": [CLASSIFICATION]},
    
#     # Conciseness
#     "length_ratio": {"name": "Length Ratio", "category": CAT_CONCISENESS, "higher_is_better": False, "explanation": "Ratio of model answer's word count to ground truth's. Closer to 1.0 often desired.", "tasks": [RAG_FAQ, SUMMARIZATION, CHATBOT]},
    
#     # Safety (Basic Checks)
#     "safety_keyword_score": {"name": "Safety Keyword Score", "category": CAT_SAFETY, "higher_is_better": True, "explanation": "Basic check: 1.0 if NO predefined unsafe keywords found, 0.0 if ANY found. Issues are flagged if score is < 1.0.", "tasks": [RAG_FAQ, SUMMARIZATION, CHATBOT]},
    
#     # Privacy/Sensitive Data (Basic Checks)
#     "pii_detection_score": {"name": "PII Detection Score", "category": CAT_PII_SAFETY, "higher_is_better": True, "explanation": "Basic regex check: 1.0 if NO common PII patterns found, 0.0 if ANY found. Issues are flagged if score is < 1.0. Not a comprehensive scan.", "tasks": [RAG_FAQ, CHATBOT]},
    
#     # Tone & Professionalism
#     "professional_tone_score": {"name": "Professional Tone", "category": CAT_TONE, "higher_is_better": True, "explanation": "Placeholder for professional tone evaluation. Currently returns NaN.", "tasks": [RAG_FAQ, CHATBOT], "status": "placeholder"},
    
#     # Refusal Appropriateness
#     "refusal_quality_score": {"name": "Refusal Quality", "category": CAT_REFUSAL, "higher_is_better": True, "explanation": "Placeholder for evaluating refusal appropriateness. Currently returns NaN.", "tasks": [RAG_FAQ, CHATBOT], "status": "placeholder"},
# }

# METRICS_BY_CATEGORY = defaultdict(list)
# CATEGORY_ORDER = [
#     CAT_TRUST, CAT_COMPLETENESS, CAT_FLUENCY, CAT_SEMANTIC, CAT_CLASSIFICATION, 
#     CAT_CONCISENESS, CAT_SAFETY, CAT_PII_SAFETY, CAT_TONE, CAT_REFUSAL
# ]
# for key, info in METRIC_INFO.items(): 
#     METRICS_BY_CATEGORY[info['category']].append(key)

# for cat_key_info in METRIC_INFO.values(): # Use .values() to iterate over dict values
#     cat = cat_key_info['category']
#     if cat not in CATEGORY_ORDER:
#         CATEGORY_ORDER.append(cat)

# REQUIRED_FIELDS_ADD_ROW = ['task_type', 'model', 'question', 'ground_truth', 'answer']
# OPTIONAL_FIELDS_ADD_ROW_INFO = {
#     "ref_facts": {
#         "label": "Reference Facts (Exact statements to find, comma-separated)", 
#         "placeholder": "e.g., The sky is blue,Earth is round", 
#         "metric_info": "For Fact Presence: Checks for these specific statements in the answer."
#     },
#     "ref_key_points": {
#         "label": "Reference Key Points/Topics (Broader concepts to cover, comma-separated)", 
#         "placeholder": "e.g., main historical events,key product features,pros and cons", 
#         "metric_info": "For Key Point Coverage: Checks if these general topics are addressed."
#     },
#     "test_description": {"label": "Test Description", "placeholder": "Briefly describe this test case's purpose...", "metric_info": "Optional metadata"}
# }

# def get_metric_display_name(metric_key, include_placeholder_tag=True):
#     info = METRIC_INFO.get(metric_key, {})
#     name = info.get('name', metric_key.replace('_', ' ').title())
#     if include_placeholder_tag and info.get("status") == "placeholder": 
#         if "(Placeholder)" not in name: 
#             name += " (Placeholder)"
#     return name

# def get_metric_indicator(metric_key):
#     info = METRIC_INFO.get(metric_key)
#     return "⬆️" if info and info["higher_is_better"] else ("⬇️" if info else "")

# def is_placeholder_metric(metric_key):
#     info = METRIC_INFO.get(metric_key, {})
#     return info.get("status") == "placeholder"

# def apply_color_gradient(styler, metric_info_dict_local):
#     cmap_good = matplotlib.colormaps.get_cmap('RdYlGn')
#     cmap_bad = matplotlib.colormaps.get_cmap('RdYlGn_r')
    
#     for col_name_display in styler.columns: 
#         if not isinstance(col_name_display, str):
#             continue
#         original_metric_key = None
#         for mk_orig, m_info_orig in metric_info_dict_local.items():
#             possible_names = [
#                 f"{get_metric_display_name(mk_orig, True)} {get_metric_indicator(mk_orig)}".strip(),
#                 f"{get_metric_display_name(mk_orig, False)} {get_metric_indicator(mk_orig)}".strip(),
#                 get_metric_display_name(mk_orig, True).strip(),
#                 get_metric_display_name(mk_orig, False).strip(),
#                 mk_orig 
#             ]
#             if col_name_display.strip() in possible_names:
#                 original_metric_key = mk_orig
#                 break
        
#         info = metric_info_dict_local.get(original_metric_key)

#         if info and pd.api.types.is_numeric_dtype(styler.data[col_name_display]) and not is_placeholder_metric(original_metric_key):
#             cmap_to_use = cmap_good if info['higher_is_better'] else cmap_bad
#             try:
#                 data_col = styler.data[col_name_display].dropna().astype(float) 
#                 if data_col.empty: 
#                     continue
#                 vmin = data_col.min()
#                 vmax = data_col.max()
                
#                 gradient_vmin = 0.0
#                 gradient_vmax = 1.0
#                 if original_metric_key == SEMANTIC_SIMILARITY_SCORE: 
#                     gradient_vmin = min(data_col.min(), -1.0) # Allow for -1 to 1 range display
#                     gradient_vmax = max(data_col.max(), 1.0)
                
#                 if np.isclose(vmin, vmax): # Handle single value columns
#                     # Determine color based on where the single value falls in the expected range
#                     # For 0-1 scores: 0 is red, 0.5 yellow, 1 green (if higher_is_better)
#                     # For -1 to 1 scores (like semantic similarity): -1 red, 0 yellow, 1 green
#                     mid_point_norm = (gradient_vmin + gradient_vmax) / 2.0
#                     # Normalize vmin to the 0-1 scale for cmap
#                     if (gradient_vmax - gradient_vmin) == 0: # Avoid division by zero
#                         norm_val = 0.5 # Default to midpoint color
#                     else:
#                         norm_val = (vmin - gradient_vmin) / (gradient_vmax - gradient_vmin)
                    
#                     color_val_single = norm_val
#                     if not info['higher_is_better']:
#                         color_val_single = 1.0 - norm_val # Invert for lower is better
                    
#                     styler.background_gradient(cmap=matplotlib.colors.ListedColormap([cmap_to_use(color_val_single)]), subset=[col_name_display])

#                 else: 
#                     styler.background_gradient(cmap=cmap_to_use, subset=[col_name_display], vmin=gradient_vmin, vmax=gradient_vmax)
#             except Exception as e:
#                 warnings.warn(f"Style error for '{col_name_display}' (orig key: {original_metric_key}): {e}", RuntimeWarning)
    
#     format_dict = {}
#     for col_disp_name_format in styler.columns: 
#         if isinstance(col_disp_name_format, str):
#             original_mkey_for_format = None
#             for mk_orig_fmt, m_info_orig_fmt in metric_info_dict_local.items():
#                 possible_names_fmt = [
#                     f"{get_metric_display_name(mk_orig_fmt, True)} {get_metric_indicator(mk_orig_fmt)}".strip(),
#                     f"{get_metric_display_name(mk_orig_fmt, False)} {get_metric_indicator(mk_orig_fmt)}".strip(),
#                     get_metric_display_name(mk_orig_fmt, True).strip(),
#                     get_metric_display_name(mk_orig_fmt, False).strip(),
#                     mk_orig_fmt
#                 ]
#                 if col_disp_name_format.strip() in possible_names_fmt:
#                     original_mkey_for_format = mk_orig_fmt
#                     break
            
#             if original_mkey_for_format and pd.api.types.is_numeric_dtype(styler.data[col_disp_name_format]):
#                  format_dict[col_disp_name_format] = '{:.4f}'
#             elif col_disp_name_format in ['Observations', 'Potential Actions', 'Metrics Not Computed or Not Applicable']: 
#                 pass 
#             elif styler.data[col_disp_name_format].dtype == 'object' and any(isinstance(x, float) for x in styler.data[col_disp_name_format].dropna()):
#                  format_dict[col_disp_name_format] = lambda x: f"{x:.4f}" if isinstance(x, float) else x

#     styler.format(formatter=format_dict, na_rep='NaN')
#     return styler

# def unflatten_df_to_test_cases(df):
#     test_cases_list = []
#     if df is None or df.empty: return []
#     direct_keys = ['id', 'task_type', 'model', 'question', 'ground_truth', 'answer', 
#                    'ref_facts', 'ref_key_points', 'test_description', 'contexts'] 
    
#     for _, row_series in df.iterrows():
#         row = row_series.to_dict()
#         case = {}
#         if pd.isna(row.get('task_type')) or pd.isna(row.get('model')) or \
#            pd.isna(row.get('question')) or pd.isna(row.get('ground_truth')) or \
#            pd.isna(row.get('answer')):
#             st.warning(f"Skipping row due to missing required field(s) (task_type, model, question, ground_truth, answer): {row.get('id', 'Unknown ID')}")
#             continue
#         for key in direct_keys:
#             if key in row and pd.notna(row[key]):
#                 case[key] = str(row[key])
#             elif key in row: 
#                 case[key] = None 
#         for col_name, value in row.items():
#             if col_name not in case: 
#                 case[col_name] = str(value) if pd.notna(value) else None
#         test_cases_list.append(case)
#     return test_cases_list

# def generate_single_case_interpretation(case_row, task_type):
#     observations = []
#     suggestions = []
#     not_applicable_metrics = [] 

#     overall_assessment_flags = { 
#         "fluency": None, "semantic": None, "factuality": None, "completeness": None, 
#         "classification": None, "conciseness": None, "safety": "ok", "privacy": "ok" # Default to ok
#     }

#     def was_input_provided(metric_key, case_data):
#         metric_detail = METRIC_INFO.get(metric_key)
#         if metric_detail and "input_field_data_key" in metric_detail:
#             input_key = metric_detail["input_field_data_key"]
#             return input_key in case_data and pd.notna(case_data[input_key]) and str(case_data[input_key]).strip()
#         return True 

#     # --- Fluency & Lexical Similarity ---
#     fluency_metric_keys = ['bleu', 'rouge_1', 'rouge_2', 'rouge_l', 'meteor']
#     valid_fluency_scores = []
#     has_any_fluency_metric = any(key in case_row for key in fluency_metric_keys)
#     for key in fluency_metric_keys:
#         if key in case_row and pd.notna(case_row[key]) and not is_placeholder_metric(key):
#             valid_fluency_scores.append(case_row[key])
#         elif key in case_row and pd.isna(case_row[key]) and not is_placeholder_metric(key):
#             not_applicable_metrics.append(f"{get_metric_display_name(key, False)}: Score is NaN.")
#     if valid_fluency_scores:
#         avg_fluency = np.mean(valid_fluency_scores)
#         if avg_fluency >= 0.6: observations.append(f"✅ Fluency & Lexical Sim.: Strong (Avg. Score: {avg_fluency:.2f})."); overall_assessment_flags["fluency"] = "good"
#         elif avg_fluency >= 0.3: observations.append(f"⚠️ Fluency & Lexical Sim.: Moderate (Avg. Score: {avg_fluency:.2f})."); overall_assessment_flags["fluency"] = "moderate"
#         else: observations.append(f"❌ Fluency & Lexical Sim.: Low (Avg. Score: {avg_fluency:.2f})."); overall_assessment_flags["fluency"] = "poor"; suggestions.append("Lexical similarity is low. Review for grammar, coherence. Compare with Semantic Similarity.")

#     # --- Semantic Understanding ---
#     metric_key_sem_sim = SEMANTIC_SIMILARITY_SCORE
#     score_sem_sim = case_row.get(metric_key_sem_sim)
#     if metric_key_sem_sim in case_row:
#         if is_placeholder_metric(metric_key_sem_sim): not_applicable_metrics.append(f"{get_metric_display_name(metric_key_sem_sim, False)}: Placeholder unexpectedly.")
#         elif pd.notna(score_sem_sim):
#             if score_sem_sim >= 0.75: observations.append(f"✅ Semantic Similarity: Strong ({score_sem_sim:.2f}). Meaning highly aligned."); overall_assessment_flags["semantic"] = "good"
#             elif score_sem_sim >= 0.5: observations.append(f"ℹ️ Semantic Similarity: Moderate ({score_sem_sim:.2f}). Meaning somewhat aligned."); overall_assessment_flags["semantic"] = "moderate"
#             elif score_sem_sim >= 0.25: observations.append(f"⚠️ Semantic Similarity: Fair ({score_sem_sim:.2f}). Some semantic overlap but may miss key aspects."); overall_assessment_flags["semantic"] = "fair"
#             else: observations.append(f"❌ Semantic Similarity: Low ({score_sem_sim:.2f}). Meaning diverges significantly."); overall_assessment_flags["semantic"] = "poor"; suggestions.append("Semantic similarity is low. Model may misunderstand query or diverge factually.")
#         elif pd.isna(score_sem_sim): 
#             sem_sim_metric_info = METRIC_INFO.get(metric_key_sem_sim, {})
#             if "sentence-transformers library" in sem_sim_metric_info.get("explanation", "").lower() or "failed to load" in sem_sim_metric_info.get("explanation", "").lower() :
#                  not_applicable_metrics.append(f"{get_metric_display_name(metric_key_sem_sim, False)}: Not computed (Sentence Transformers library/model issue).")
#             else: not_applicable_metrics.append(f"{get_metric_display_name(metric_key_sem_sim, False)}: Score is NaN.")

#     # --- Trust & Factuality (Fact Presence) ---
#     metric_key_fp = 'fact_presence_score'
#     score_fp = case_row.get(metric_key_fp)
#     if metric_key_fp in case_row:
#         if not was_input_provided(metric_key_fp, case_row): not_applicable_metrics.append(f"{get_metric_display_name(metric_key_fp, False)}: Not applicable (missing `ref_facts`).")
#         elif pd.notna(score_fp):
#             if score_fp >= 0.7: observations.append(f"✅ Fact Presence: Good ({score_fp:.2f}). Most specified facts included."); overall_assessment_flags["factuality"] = "good"
#             elif score_fp >= 0.4: observations.append(f"⚠️ Fact Presence: Moderate ({score_fp:.2f}). Some facts missing/altered."); overall_assessment_flags["factuality"] = "moderate"
#             else: observations.append(f"❌ Fact Presence: Low ({score_fp:.2f}). Significant facts missing."); overall_assessment_flags["factuality"] = "poor"; suggestions.append("Verify missing/inaccurate critical facts from `ref_facts`.")
#         elif pd.isna(score_fp): not_applicable_metrics.append(f"{get_metric_display_name(metric_key_fp, False)}: Score is NaN (Input `ref_facts` provided; review calculation).")

#     # --- Completeness & Coverage (Key Point Coverage) ---
#     metric_key_cc = 'completeness_score'
#     score_cc = case_row.get(metric_key_cc)
#     if metric_key_cc in case_row:
#         if not was_input_provided(metric_key_cc, case_row): not_applicable_metrics.append(f"{get_metric_display_name(metric_key_cc, False)}: Not applicable (missing `ref_key_points`).")
#         elif pd.notna(score_cc):
#             if score_cc >= 0.7: observations.append(f"✅ Key Point Coverage: Good ({score_cc:.2f}). Most key points covered."); overall_assessment_flags["completeness"] = "good"
#             elif score_cc >= 0.4: observations.append(f"⚠️ Key Point Coverage: Moderate ({score_cc:.2f}). Some key points unaddressed."); overall_assessment_flags["completeness"] = "moderate"
#             else: observations.append(f"❌ Key Point Coverage: Low ({score_cc:.2f}). Significant key points missing."); overall_assessment_flags["completeness"] = "poor"; suggestions.append("Verify if essential topics from `ref_key_points` were addressed.")
#         elif pd.isna(score_cc): not_applicable_metrics.append(f"{get_metric_display_name(metric_key_cc, False)}: Score is NaN (Input `ref_key_points` provided; review calculation).")

#     # --- Classification Accuracy ---
#     if task_type == CLASSIFICATION:
#         accuracy = case_row.get('accuracy')
#         if 'accuracy' in case_row and pd.notna(accuracy): # Assuming accuracy is always computed if task is CLASSIFICATION
#             if accuracy < 1.0: observations.append(f"❌ Classification: Incorrect (Accuracy: {accuracy:.2f})."); overall_assessment_flags["classification"] = "incorrect"; suggestions.append("Analyze misclassification. Review input/ground truth.")
#             else: observations.append(f"✅ Classification: Correct (Accuracy: {accuracy:.2f})."); overall_assessment_flags["classification"] = "correct"
#         elif 'accuracy' in case_row and pd.isna(accuracy): not_applicable_metrics.append(f"{get_metric_display_name('accuracy', False)}: Score is NaN.")
    
#     # --- Conciseness ---
#     length_ratio = case_row.get('length_ratio')
#     if 'length_ratio' in case_row and pd.notna(length_ratio):
#         if length_ratio < 0.5: observations.append(f"⚠️ Conciseness: Response significantly shorter (ratio: {length_ratio:.2f})."); overall_assessment_flags["conciseness"] = "too_short"; suggestions.append("Check if response is overly brief/truncated.")
#         elif length_ratio < 0.8: observations.append(f"ℹ️ Conciseness: Response noticeably shorter (ratio: {length_ratio:.2f})."); overall_assessment_flags["conciseness"] = "short"
#         elif length_ratio <= 1.25: observations.append(f"✅ Conciseness: Response length comparable to reference (ratio: {length_ratio:.2f})."); overall_assessment_flags["conciseness"] = "good"
#         elif length_ratio <= 1.75: observations.append(f"ℹ️ Conciseness: Response noticeably longer (ratio: {length_ratio:.2f})."); overall_assessment_flags["conciseness"] = "long"
#         else: observations.append(f"⚠️ Conciseness: Response significantly longer (ratio: {length_ratio:.2f})."); overall_assessment_flags["conciseness"] = "too_long"; suggestions.append("Check if response is too verbose/irrelevant.")
#     elif 'length_ratio' in case_row and pd.isna(length_ratio): not_applicable_metrics.append(f"{get_metric_display_name('length_ratio', False)}: Score is NaN.")

#     # --- Safety & Privacy (Only show if issue detected) ---
#     safety_score = case_row.get('safety_keyword_score')
#     if 'safety_keyword_score' in case_row and pd.notna(safety_score):
#         if safety_score < 1.0: 
#             observations.append("🚨 Safety: Potential safety keyword detected (Basic Check).")
#             overall_assessment_flags["safety"] = "issue"
#             suggestions.append("MANUAL REVIEW REQUIRED for safety. Identify problematic content; refine safety filters/prompts.")
#     elif 'safety_keyword_score' in case_row and pd.isna(safety_score): 
#         not_applicable_metrics.append(f"{get_metric_display_name('safety_keyword_score', False)}: Score is NaN.")

#     pii_score = case_row.get('pii_detection_score')
#     if 'pii_detection_score' in case_row and pd.notna(pii_score):
#         if pii_score < 1.0: 
#             observations.append("🚨 Privacy: Potential PII pattern detected (Basic Regex Check).")
#             overall_assessment_flags["privacy"] = "issue"
#             suggestions.append("MANUAL REVIEW REQUIRED for PII. Ensure sensitive data is not exposed; enhance PII detection/scrubbing.")
#     elif 'pii_detection_score' in case_row and pd.isna(pii_score): 
#         not_applicable_metrics.append(f"{get_metric_display_name('pii_detection_score', False)}: Score is NaN.")
        
#     # Handle placeholder metrics
#     placeholder_keys_to_check = ["professional_tone_score", "refusal_quality_score", "nli_entailment_score", "llm_judge_factuality"]
#     for pk in placeholder_keys_to_check:
#         if pk in case_row: 
#             metric_display_name_pk = get_metric_display_name(pk, False) 
#             if is_placeholder_metric(pk):
#                 metric_explanation = METRIC_INFO.get(pk, {}).get('explanation', 'Full implementation pending.')
#                 if pd.isna(case_row[pk]): 
#                     not_applicable_metrics.append(f"{metric_display_name_pk}: Placeholder - Not implemented. ({metric_explanation})")
#                 else: 
#                     not_applicable_metrics.append(f"{metric_display_name_pk}: Placeholder received unexpected score ({case_row[pk]:.2f}).")

#     # --- Final Summary for Suggestions ---
#     # (This logic can remain largely the same as in previous full version)
#     final_suggestions_summary_parts = []
#     issues_found_text = []
#     positives_found_text = []

#     if overall_assessment_flags["fluency"] == "poor": issues_found_text.append("low fluency/lexical similarity")
#     elif overall_assessment_flags["fluency"] == "good": positives_found_text.append("good fluency/lexical similarity")
#     if overall_assessment_flags["semantic"] == "poor": issues_found_text.append("low semantic similarity")
#     elif overall_assessment_flags["semantic"] == "good": positives_found_text.append("good semantic similarity")
#     if overall_assessment_flags["factuality"] == "poor": issues_found_text.append("low fact presence")
#     elif overall_assessment_flags["factuality"] == "good": positives_found_text.append("good fact presence")
#     if overall_assessment_flags["completeness"] == "poor": issues_found_text.append("low key point coverage")
#     elif overall_assessment_flags["completeness"] == "good": positives_found_text.append("good key point coverage")
#     if overall_assessment_flags["classification"] == "incorrect": issues_found_text.append("incorrect classification")
#     elif overall_assessment_flags["classification"] == "correct": positives_found_text.append("correct classification")
#     if overall_assessment_flags["conciseness"] == "too_short": issues_found_text.append("response too short")
#     elif overall_assessment_flags["conciseness"] == "too_long": issues_found_text.append("response too long")
#     elif overall_assessment_flags["conciseness"] == "good": positives_found_text.append("appropriate length")
#     if overall_assessment_flags["safety"] == "issue": issues_found_text.append("potential safety concerns")
#     if overall_assessment_flags["privacy"] == "issue": issues_found_text.append("potential PII exposure")

#     if issues_found_text:
#         summary_statement = f"Overall, potential concerns regarding: {', '.join(issues_found_text)}. "
#         if positives_found_text: summary_statement += f"However, performed well in: {', '.join(positives_found_text)}. "
#         summary_statement += "Consider specific suggestions."
#         final_suggestions_summary_parts.append(summary_statement)
#     elif positives_found_text:
#         final_suggestions_summary_parts.append(f"Overall, performed well regarding: {', '.join(positives_found_text)}. No major issues flagged by automated metrics.")
#     else: 
#         if not observations and not_applicable_metrics: final_suggestions_summary_parts.append("Most metrics not applicable/computed. Manual review needed for specific evaluations.")
#         elif not observations : final_suggestions_summary_parts.append("No specific issues/strengths flagged by automated metrics. Manual review recommended for nuanced aspects.")

#     final_suggestions_text = "\n".join(f"- {s_item}" for s_item in final_suggestions_summary_parts + suggestions if s_item)
#     if not final_suggestions_text.strip(): final_suggestions_text = "- Review case manually for unmeasured criteria."

#     if not observations and not_applicable_metrics: observations.append("No specific metric observations (scores might be NaN, metrics not applicable, or placeholders). See 'Metrics Not Computed or Not Applicable'.")
#     elif not observations: observations.append("No specific metric observations generated. Scores might be within acceptable ranges or metrics not applicable for this task.")
    
#     return "\n".join(f"- {o}" for o in observations), final_suggestions_text, "\n".join(f"- {na}" for na in not_applicable_metrics if na)


# # --- Streamlit App Setup ---
# st.set_page_config(layout="wide", page_title="LLM Evaluation Framework")
# st.title("📊 LLM Evaluation Framework")
# st.markdown("Evaluate LLM performance using pre-generated responses. This tool supports aggregated summaries and individual test case scores, including lexical and semantic similarity.")

# default_state_keys = {
#     'test_cases_list_loaded': None, 'edited_test_cases_df': pd.DataFrame(),
#     'aggregated_results_df': None, 'individual_scores_df': None, 
#     'data_source_info': None, 'last_uploaded_file_name': None,
#     'metrics_for_agg_display': [],
#     'add_row_input_mode': "Easy (Required Fields Only)" 
# }
# for key, default_value in default_state_keys.items():
#     if key not in st.session_state:
#         st.session_state[key] = copy.deepcopy(default_value)

# st.sidebar.header("⚙️ Input Options")
# def clear_app_state():
#     st.session_state.test_cases_list_loaded = None
#     st.session_state.edited_test_cases_df = pd.DataFrame()
#     st.session_state.aggregated_results_df = None
#     st.session_state.individual_scores_df = None
#     st.session_state.data_source_info = None
#     st.session_state.metrics_for_agg_display = []
#     st.session_state.last_uploaded_file_name = None 

# input_method = st.sidebar.radio(
#     "Choose data source:", ("Upload File", "Generate Mock Data"),
#     key="input_method_radio", on_change=clear_app_state
# )

# if input_method == "Upload File":
#     uploaded_file = st.sidebar.file_uploader(
#         "Upload (.xlsx, .csv, .json - Flat Format)", type=["xlsx", "csv", "json"],
#         key="file_uploader_widget" 
#     )
#     if uploaded_file is not None:
#         if uploaded_file.name != st.session_state.last_uploaded_file_name: 
#             clear_app_state() 
#             st.session_state.last_uploaded_file_name = uploaded_file.name
#             file_suffix = Path(uploaded_file.name).suffix.lower()
#             st.session_state.data_source_info = f"Processing: {uploaded_file.name}"
#             st.sidebar.info(st.session_state.data_source_info)
#             tmp_file_path = None
#             try:
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
#                     tmp_file.write(uploaded_file.getvalue())
#                     tmp_file_path = Path(tmp_file.name)
#                 test_data_list_from_file = None
#                 with st.spinner(f"Loading and converting {uploaded_file.name}..."):
#                     if file_suffix == ".xlsx": test_data_list_from_file = convert_excel_to_data(tmp_file_path)
#                     elif file_suffix == ".csv": test_data_list_from_file = convert_csv_to_data(tmp_file_path)
#                     elif file_suffix == ".json": test_data_list_from_file = load_data(tmp_file_path)
#                 if test_data_list_from_file:
#                     st.session_state.test_cases_list_loaded = test_data_list_from_file
#                     df_for_edit = pd.DataFrame(test_data_list_from_file)
#                     required_cols_editor = ['id', 'task_type', 'model', 'question', 'ground_truth', 'answer', 
#                                             'ref_facts', 'ref_key_points', 'test_description', 'contexts'] 
#                     for col in required_cols_editor:
#                         if col not in df_for_edit.columns: df_for_edit[col] = None 
#                     st.session_state.edited_test_cases_df = df_for_edit.copy().fillna('')
#                     st.session_state.data_source_info = f"Loaded {len(test_data_list_from_file)} rows from {uploaded_file.name} into editor."
#                     st.sidebar.success(st.session_state.data_source_info)
#                 else:
#                     if test_data_list_from_file == []: raise ValueError("File loaded but was empty or contained no valid data rows.")
#                     else: raise ValueError("Failed to load/convert data. Check file format, content, and required columns.")
#             except Exception as e:
#                 st.session_state.data_source_info = f"Error processing {uploaded_file.name}: {e}"
#                 st.sidebar.error(st.session_state.data_source_info); 
#                 st.sidebar.text_area("Traceback", traceback.format_exc(), height=150)
#                 clear_app_state() 
#             finally:
#                 if tmp_file_path and tmp_file_path.exists():
#                     try: os.unlink(tmp_file_path)
#                     except Exception as e_unlink: warnings.warn(f"Could not delete temp file {tmp_file_path}: {e_unlink}")
#         elif st.session_state.data_source_info: st.sidebar.info(st.session_state.data_source_info)
# elif input_method == "Generate Mock Data":
#     st.sidebar.warning("Mock data provides example flat-format rows with varied answer quality.")
#     if st.sidebar.button("Generate and Use Mock Data", key="generate_mock_button"):
#         clear_app_state()
#         try:
#             with st.spinner("Generating mock evaluation data..."):
#                 mock_data_list = generate_mock_data_flat(num_samples_per_task=3) 
#             if mock_data_list:
#                 st.session_state.test_cases_list_loaded = mock_data_list
#                 df_for_edit = pd.DataFrame(mock_data_list)
#                 required_cols_editor = ['id', 'task_type', 'model', 'question', 'ground_truth', 'answer', 
#                                         'ref_facts', 'ref_key_points', 'test_description', 'contexts']
#                 for col in required_cols_editor:
#                     if col not in df_for_edit.columns: df_for_edit[col] = None
#                 st.session_state.edited_test_cases_df = df_for_edit.copy().fillna('')
#                 st.session_state.data_source_info = f"Using {len(mock_data_list)} generated mock rows, loaded into editor."
#                 st.sidebar.success(st.session_state.data_source_info)
#             else: st.sidebar.error("Failed to generate mock data.")
#         except Exception as e:
#             st.sidebar.error(f"Error generating mock data: {e}"); st.sidebar.text_area("Traceback", traceback.format_exc(), height=150); clear_app_state()

# if st.session_state.data_source_info:
#     if "error" in st.session_state.data_source_info.lower() or "failed" in st.session_state.data_source_info.lower() : st.error(st.session_state.data_source_info)
#     elif "loaded" in st.session_state.data_source_info.lower() or "using" in st.session_state.data_source_info.lower(): st.success(st.session_state.data_source_info)
#     else: st.info(st.session_state.data_source_info)

# tab_eval, tab_data_editor, tab_format_guide, tab_metrics_tutorial = st.tabs(["📊 Evaluation & Results", "📝 View/Edit/Add Data", "📄 Data Format Guide", "📖 Metrics Tutorial"])

# with tab_eval:
#     st.header("Run Evaluation and View Results")
#     run_button_disabled = not isinstance(st.session_state.edited_test_cases_df, pd.DataFrame) or st.session_state.edited_test_cases_df.empty
#     st.info("ℹ️ **Semantic Similarity Note:** First run may download model files (requires internet). Subsequent runs use cache. For offline use, pre-download model (see tool output/docs).")
#     if st.button("🚀 Run Evaluation on Data in Editor", disabled=run_button_disabled, key="run_eval_main_button", help="Evaluates data from 'View/Edit/Add Data' tab."):
#         # ... (Evaluation logic - largely same as genai_eval_tool_streamlit_app_may16_full, ensure it uses the updated generate_single_case_interpretation)
#         if isinstance(st.session_state.edited_test_cases_df, pd.DataFrame) and not st.session_state.edited_test_cases_df.empty:
#             st.session_state.aggregated_results_df = None; st.session_state.individual_scores_df = None
#             st.session_state.metrics_for_agg_display = []
#             with st.spinner("⏳ Evaluating... This may take a moment, especially if downloading models for semantic similarity."):
#                 try:
#                     df_to_process = st.session_state.edited_test_cases_df.replace('', np.nan) 
#                     test_cases_to_evaluate = unflatten_df_to_test_cases(df_to_process)
#                     if not test_cases_to_evaluate: raise ValueError("Data in editor is empty or could not be processed into valid test cases. Ensure required fields are filled.")
                    
#                     individual_df_raw, aggregated_df = evaluate_model_responses(test_cases_to_evaluate)
                    
#                     if individual_df_raw is not None and not individual_df_raw.empty:
#                         interpretations_output = individual_df_raw.apply(
#                             lambda row: generate_single_case_interpretation(row, row.get('task_type')), axis=1
#                         )
#                         individual_df_raw['Observations'] = interpretations_output.apply(lambda x: x[0])
#                         individual_df_raw['Potential Actions'] = interpretations_output.apply(lambda x: x[1])
#                         individual_df_raw['Metrics Not Computed or Not Applicable'] = interpretations_output.apply(lambda x: x[2])
                    
#                     st.session_state.individual_scores_df = individual_df_raw 
#                     st.session_state.aggregated_results_df = aggregated_df

#                     if aggregated_df is not None and not aggregated_df.empty:
#                         metrics_present_in_agg = [col for col in aggregated_df.columns if col in METRIC_INFO] 
#                         metrics_to_show_agg = []
#                         for metric_col in metrics_present_in_agg: 
#                             if pd.api.types.is_numeric_dtype(aggregated_df[metric_col]):
#                                 if not (aggregated_df[metric_col].isna() | (aggregated_df[metric_col].abs() < 1e-9)).all() and not is_placeholder_metric(metric_col): 
#                                     metrics_to_show_agg.append(metric_col)
#                         st.session_state.metrics_for_agg_display = metrics_to_show_agg 
#                         st.success("✅ Evaluation complete! View results below.")
#                     elif individual_df_raw is not None and not individual_df_raw.empty:
#                          st.warning("⚠️ Evaluation produced individual scores, but aggregated results are empty or only contain placeholders/NaNs.")
#                     else: st.warning("⚠️ Evaluation finished, but no results produced. Check input data and console logs.")
#                 except Exception as e:
#                      st.error(f"Evaluation error: {e}"); st.error(f"Traceback: {traceback.format_exc()}")
#         else: st.warning("No data in editor. Load or generate data first.")

#     st.divider(); st.header("Evaluation Results")
#     res_tab_ind, res_tab_agg = st.tabs(["📄 Individual Scores","📈 Aggregated Results"])

#     with res_tab_agg:
#         # ... (Aggregated results display - largely same as genai_eval_tool_streamlit_app_may16_full) ...
#         st.markdown("Aggregated view transforms individual data points into insights about the LLM's overall behavior. It helps identify strengths, weaknesses, and potential biases, informing decisions on model development and deployment.")
#         if st.session_state.aggregated_results_df is not None and not st.session_state.aggregated_results_df.empty:
#             agg_df = st.session_state.aggregated_results_df
#             metrics_to_display_non_placeholder = st.session_state.metrics_for_agg_display
#             if not metrics_to_display_non_placeholder:
#                 st.info("No non-placeholder metrics with valid scores to display in aggregated summary. Displaying raw data if available.")
#                 simple_formatter = {col: "{:.4f}" for col in agg_df.select_dtypes(include=np.number).columns}
#                 st.dataframe(agg_df.style.format(formatter=simple_formatter, na_rep='NaN'), use_container_width=True)
#             else:
#                 st.markdown("#### 🏆 Best Model Summary (Highlights)")
#                 # ... (Best model summary logic - unchanged)
#                 st.caption("Top models per task based on key, non-placeholder metrics (non-zero/NaN scores only). Expand tasks.")
#                 key_metrics_per_task = {
#                     RAG_FAQ: [m for m in [SEMANTIC_SIMILARITY_SCORE, "fact_presence_score", "completeness_score", "rouge_l"] if not is_placeholder_metric(m)],
#                     SUMMARIZATION: [m for m in [SEMANTIC_SIMILARITY_SCORE, "completeness_score", "rouge_l", "length_ratio"] if not is_placeholder_metric(m)],
#                     CLASSIFICATION: [m for m in ["f1_score", "accuracy"] if not is_placeholder_metric(m)],
#                     CHATBOT: [m for m in [SEMANTIC_SIMILARITY_SCORE, "meteor", "rouge_l", "length_ratio"] if not is_placeholder_metric(m)]
#                 }
#                 available_tasks_for_best_model = sorted(agg_df['task_type'].unique()) if 'task_type' in agg_df else []
#                 if not available_tasks_for_best_model: st.info("Run evaluation to see best model summaries.")
#                 else:
#                     for task_type_bm in available_tasks_for_best_model:
#                         with st.expander(f"**Task: {task_type_bm}**", expanded=False):
#                             task_df_bm = agg_df[agg_df['task_type'] == task_type_bm].copy()
#                             best_performers_details = []
#                             current_task_key_metrics = [
#                                 m for m in key_metrics_per_task.get(task_type_bm, []) 
#                                 if m in task_df_bm.columns and pd.api.types.is_numeric_dtype(task_df_bm[m]) and m in metrics_to_display_non_placeholder
#                             ]
#                             if not current_task_key_metrics:
#                                 st.markdown("_No key, non-placeholder metrics available for highlights in this task._")
#                                 continue
#                             for metric_bm in current_task_key_metrics:
#                                 info_bm = METRIC_INFO.get(metric_bm, {})
#                                 if not info_bm : continue 
#                                 higher_better_bm = info_bm.get('higher_is_better', True)
#                                 valid_scores_df_bm = task_df_bm.dropna(subset=[metric_bm])
#                                 if valid_scores_df_bm.empty or (valid_scores_df_bm[metric_bm].abs() < 1e-9).all(): continue 
                                
#                                 best_score_idx_bm = valid_scores_df_bm[metric_bm].idxmax() if higher_better_bm else valid_scores_df_bm[metric_bm].idxmin()
#                                 best_row_bm = valid_scores_df_bm.loc[best_score_idx_bm]
#                                 best_model_bm = best_row_bm['model']
#                                 best_score_val_bm = best_row_bm[metric_bm]
#                                 if pd.notna(best_score_val_bm) and not np.isclose(best_score_val_bm, 0.0, atol=1e-9): 
#                                     best_performers_details.append({
#                                         "metric_name_display": f"{get_metric_display_name(metric_bm, False)} {get_metric_indicator(metric_bm)}", 
#                                         "model": best_model_bm, "score": best_score_val_bm,
#                                         "explanation": info_bm.get('explanation', 'N/A')})
#                             if not best_performers_details: st.markdown("_No significant highlights determined for this task (all key metric scores might be zero or NaN)._")
#                             else:
#                                 highlights_by_model = defaultdict(list)
#                                 for detail in best_performers_details: highlights_by_model[detail["model"]].append(detail)
#                                 for model_name_bm, model_highlights in sorted(highlights_by_model.items()):
#                                     st.markdown(f"**`{model_name_bm}` was best for:**")
#                                     for highlight in model_highlights:
#                                         st.markdown(f"- {highlight['metric_name_display']}: **{highlight['score']:.4f}**")
#                                         st.caption(f"  *Metric Meaning:* {highlight['explanation']}")
#                                     st.markdown("---") 
#                     st.markdown("---") 
#                 st.markdown("#### 📊 Overall Summary Table (Aggregated by Task & Model)")
#                 # ... (Overall summary table logic - unchanged)
#                 agg_df_display_overall = agg_df.copy()
#                 renamed_cols_overall = {}
#                 final_display_cols_overall = []
#                 static_cols_display = ['task_type', 'model', 'num_samples']
#                 for static_col in static_cols_display:
#                     if static_col in agg_df_display_overall.columns:
#                         new_name = static_col.replace('_', ' ').title()
#                         renamed_cols_overall[static_col] = new_name
#                         final_display_cols_overall.append(new_name)
#                 for original_metric_key in metrics_to_display_non_placeholder: 
#                     if original_metric_key in agg_df_display_overall.columns: 
#                         indicator = get_metric_indicator(original_metric_key)
#                         col_title = get_metric_display_name(original_metric_key, include_placeholder_tag=False) 
#                         new_name = f"{col_title} {indicator}".strip()
#                         renamed_cols_overall[original_metric_key] = new_name
#                         final_display_cols_overall.append(new_name)
#                 agg_df_display_overall.rename(columns=renamed_cols_overall, inplace=True)
#                 final_display_cols_overall = [col for col in final_display_cols_overall if col in agg_df_display_overall.columns]
#                 formatter_overall = {}
#                 for original_key in metrics_to_display_non_placeholder: 
#                     renamed_key_val = renamed_cols_overall.get(original_key) 
#                     if renamed_key_val and renamed_key_val in final_display_cols_overall and \
#                        original_key in agg_df.columns and pd.api.types.is_numeric_dtype(agg_df[original_key]):
#                         formatter_overall[renamed_key_val] = "{:.4f}" 
#                 st.dataframe(
#                     agg_df_display_overall[final_display_cols_overall].style.format(formatter=formatter_overall, na_rep='NaN'),
#                     use_container_width=True
#                 )
#                 st.markdown("---")
#                 st.markdown("#### 🔍 Interpreting Your Aggregated Results (Experimental)")
#                 # ... (Aggregated interpretation logic - unchanged, uses the updated generate_single_case_interpretation indirectly via overall flags)
#                 with st.expander("💡 Interpreting Your Aggregated Results (Experimental)", expanded=False):
#                     st.markdown("""
#                     This section offers a general interpretation of the aggregated scores for non-placeholder metrics. 
#                     Remember that these are heuristic-based and should be combined with a qualitative review of individual responses.
#                     Low scores don't always mean a "bad" model; they indicate areas where the model's output differs from the reference or desired behavior according to the specific metric. 
#                     NaN scores indicate the metric was not applicable (e.g., missing input for `fact_presence_score`) or could not be computed.
#                     """)
#                     if agg_df is not None and not agg_df.empty:
#                         for task_type_interp in agg_df['task_type'].unique():
#                             st.markdown(f"#### Task: {task_type_interp}")
#                             task_data_interp = agg_df[agg_df['task_type'] == task_type_interp]
#                             for model_name_interp in task_data_interp['model'].unique():
#                                 model_scores_interp = task_data_interp[task_data_interp['model'] == model_name_interp].iloc[0]
#                                 st.markdown(f"**Model: `{model_name_interp}`**")
#                                 interpretations = []
#                                 suggestions_interp = [] 

#                                 fluency_scores_interp = {m: model_scores_interp.get(m) for m in ['bleu', 'rouge_l', 'meteor'] if m in model_scores_interp and pd.notna(model_scores_interp.get(m)) and not is_placeholder_metric(m)}
#                                 if fluency_scores_interp:
#                                     valid_fluency_scores_interp = [s for s in fluency_scores_interp.values() if pd.notna(s)]
#                                     if valid_fluency_scores_interp:
#                                         avg_fluency_interp = np.mean(valid_fluency_scores_interp)
#                                         if avg_fluency_interp > 0.5: interpretations.append(f"✅ Generally good fluency & lexical similarity (Avg. relevant score: {avg_fluency_interp:.2f}).")
#                                         elif avg_fluency_interp > 0.2: interpretations.append(f"⚠️ Moderate fluency/lexical similarity. Responses may differ noticeably from references (Avg. score: {avg_fluency_interp:.2f}).")
#                                         else: interpretations.append(f"❌ Low fluency/lexical similarity. Responses might be quite different or have linguistic issues (Avg. score: {avg_fluency_interp:.2f})."); suggestions_interp.append("Review responses for clarity, grammar. Compare with semantic similarity.")
                                
#                                 semantic_sim_interp = model_scores_interp.get(SEMANTIC_SIMILARITY_SCORE)
#                                 if pd.notna(semantic_sim_interp) and not is_placeholder_metric(SEMANTIC_SIMILARITY_SCORE):
#                                     if semantic_sim_interp > 0.7: interpretations.append(f"✅ Good semantic similarity to references ({semantic_sim_interp:.2f}). Meaning is well-aligned.")
#                                     elif semantic_sim_interp > 0.4: interpretations.append(f"ℹ️ Moderate semantic similarity ({semantic_sim_interp:.2f}). Meaning somewhat aligned.")
#                                     else: interpretations.append(f"⚠️ Low semantic similarity ({semantic_sim_interp:.2f}). Meaning may differ significantly."); suggestions_interp.append("Low semantic similarity can indicate misunderstanding of query or factual divergence, even with good lexical scores.")
                                
#                                 fact_presence_interp = model_scores_interp.get('fact_presence_score')
#                                 if pd.notna(fact_presence_interp) and not is_placeholder_metric('fact_presence_score'): 
#                                     if fact_presence_interp > 0.7: interpretations.append(f"✅ Good inclusion of specified facts ({fact_presence_interp:.2f}).")
#                                     elif fact_presence_interp > 0.4: interpretations.append(f"⚠️ Moderate inclusion of facts ({fact_presence_interp:.2f}). Some info might be missing.")
#                                     else: interpretations.append(f"❌ Low inclusion of specified facts ({fact_presence_interp:.2f})."); suggestions_interp.append("Ensure `ref_facts` are accurate. Model might need better prompting for fact extraction.")
#                                 elif pd.isna(fact_presence_interp) and 'fact_presence_score' in model_scores_interp and not is_placeholder_metric('fact_presence_score'):
#                                     interpretations.append(f"ℹ️ Fact Presence: Not applicable or not computed (score is NaN). Likely due to missing `ref_facts` in input data for this model/task's test cases.")

#                                 completeness_interp = model_scores_interp.get('completeness_score')
#                                 if pd.notna(completeness_interp) and not is_placeholder_metric('completeness_score'):
#                                     if completeness_interp > 0.7: interpretations.append(f"✅ Good coverage of key points ({completeness_interp:.2f}).")
#                                     elif completeness_interp > 0.4: interpretations.append(f"⚠️ Moderate coverage of key points ({completeness_interp:.2f}). May not address all aspects.")
#                                     else: interpretations.append(f"❌ Low coverage of key points ({completeness_interp:.2f})."); suggestions_interp.append("Ensure `ref_key_points` are well-defined. Model might need prompting for comprehensiveness.")
#                                 elif pd.isna(completeness_interp) and 'completeness_score' in model_scores_interp and not is_placeholder_metric('completeness_score'):
#                                      interpretations.append(f"ℹ️ Key Point Coverage: Not applicable or not computed (score is NaN). Likely due to missing `ref_key_points` in input data.")
                                
#                                 f1_interp = model_scores_interp.get('f1_score'); acc_interp = model_scores_interp.get('accuracy')
#                                 if pd.notna(f1_interp) and not is_placeholder_metric('f1_score'): 
#                                     if f1_interp > 0.75: interpretations.append(f"✅ Good classification performance (F1: {f1_interp:.2f}).")
#                                     elif f1_interp > 0.5: interpretations.append(f"⚠️ Moderate classification performance (F1: {f1_interp:.2f}).")
#                                     else: interpretations.append(f"❌ Low classification performance (F1: {f1_interp:.2f})."); suggestions_interp.append("Review misclassified examples. Consider more/better training data, feature engineering, or model architecture for classification tasks.")
#                                 elif pd.notna(acc_interp) and not is_placeholder_metric('accuracy'): 
#                                      if acc_interp > 0.75: interpretations.append(f"✅ Good classification accuracy ({acc_interp:.2f}).")
#                                      else: interpretations.append(f"⚠️ Classification accuracy is {acc_interp:.2f}. Consider checking precision/recall/F1."); suggestions_interp.append("Review misclassified examples if accuracy is low.")
                                
#                                 length_ratio_interp = model_scores_interp.get('length_ratio')
#                                 if pd.notna(length_ratio_interp) and not is_placeholder_metric('length_ratio'):
#                                     if 0.75 <= length_ratio_interp <= 1.25: interpretations.append(f"✅ Good response length relative to reference (Ratio: {length_ratio_interp:.2f}).")
#                                     elif length_ratio_interp < 0.5: interpretations.append(f"⚠️ Responses may be too short (Ratio: {length_ratio_interp:.2f})."); suggestions_interp.append("Check if model is truncating answers or being overly brief. Adjust max tokens or prompt for more detail.")
#                                     elif length_ratio_interp > 1.75: interpretations.append(f"⚠️ Responses may be too verbose (Ratio: {length_ratio_interp:.2f})."); suggestions_interp.append("Model might be adding unnecessary information. Prompt for conciseness or set stricter length limits.")
#                                     else: interpretations.append(f"ℹ️ Length ratio is {length_ratio_interp:.2f}. Assess if appropriate for the task.")
                                
#                                 safety_score_interp = model_scores_interp.get('safety_keyword_score')
#                                 if pd.notna(safety_score_interp) and not is_placeholder_metric('safety_keyword_score'):
#                                     if safety_score_interp < 1.0: interpretations.append(f"🚨 Safety alert! Basic keyword check failed for some responses (Avg Score: {safety_score_interp:.2f}). MANUAL REVIEW OF INDIVIDUAL CASES IS CRITICAL."); suggestions_interp.append("Implement stricter content filtering. Review prompts leading to unsafe content.")
#                                     # No "else" for "passed" as per user request to only show issues
                                
#                                 pii_score_interp = model_scores_interp.get('pii_detection_score')
#                                 if pd.notna(pii_score_interp) and not is_placeholder_metric('pii_detection_score'):
#                                     if pii_score_interp < 1.0: interpretations.append(f"🚨 Privacy alert! Basic PII pattern check failed for some responses (Avg Score: {pii_score_interp:.2f}). MANUAL REVIEW IS CRITICAL."); suggestions_interp.append("Enhance PII detection/scrubbing. Review data handling policies.")
#                                     # No "else" for "passed"

#                                 if interpretations:
#                                     st.markdown("**Observations:**")
#                                     for o_item in interpretations: st.markdown(f"- {o_item}") 
#                                 if suggestions_interp: 
#                                     st.markdown("**Summary / Potential Actions:**")
#                                     for s_item in suggestions_interp: st.markdown(f"- {s_item}") 
#                                 if not interpretations and not suggestions_interp:
#                                     st.markdown("_No specific interpretations generated based on available non-placeholder scores for this model/task combination._")
#                                 st.markdown("---") 
#                     else: st.info("Run an evaluation to see interpretations.")
#                 st.markdown("---") 
#                 st.markdown("#### 📊 Task Specific Metric Table & Chart 📈  ")
#                 # ... (Task specific table & chart - unchanged)
#                 available_tasks_agg = sorted(agg_df['task_type'].unique()) if 'task_type' in agg_df else []
#                 if not available_tasks_agg: st.info("No tasks found in aggregated results.")
#                 else:
#                     task_tabs_agg = st.tabs([f"Task: {task}" for task in available_tasks_agg])
#                     for i_task_tab, task_type_tab in enumerate(available_tasks_agg):
#                         with task_tabs_agg[i_task_tab]:
#                             task_df_agg = agg_df[agg_df['task_type'] == task_type_tab].copy()
#                             task_specific_metrics_for_task_dim_view = [
#                                 m for m in get_metrics_for_task(task_type_tab) 
#                                 if m in agg_df.columns and m in metrics_to_display_non_placeholder 
#                             ]
#                             if not task_specific_metrics_for_task_dim_view:
#                                 st.info(f"No relevant, non-placeholder metrics with valid scores to display for task '{task_type_tab}'.")
#                                 continue

#                             relevant_categories_agg = sorted(list(set(METRIC_INFO[m]['category'] for m in task_specific_metrics_for_task_dim_view if m in METRIC_INFO)))
#                             ordered_relevant_categories_agg = [cat for cat in CATEGORY_ORDER if cat in relevant_categories_agg]

#                             if not ordered_relevant_categories_agg: st.info(f"No metric categories with displayable metrics for task '{task_type_tab}'.")
#                             else:
#                                 dimension_tabs_agg = st.tabs([f"{cat}" for cat in ordered_relevant_categories_agg])
#                                 for j_dim, category in enumerate(ordered_relevant_categories_agg):
#                                     with dimension_tabs_agg[j_dim]:
#                                         metrics_in_category_task_agg = [m for m in task_specific_metrics_for_task_dim_view if METRIC_INFO.get(m, {}).get('category') == category]
#                                         if not metrics_in_category_task_agg : 
#                                             st.write(f"_No '{category}' metrics available or selected for display in this task._")
#                                             continue
                                        
#                                         cols_to_show_agg_dim = ['model', 'num_samples'] + metrics_in_category_task_agg
#                                         cols_to_show_present_agg_dim = [c for c in cols_to_show_agg_dim if c in task_df_agg.columns]
                                        
#                                         st.markdown(f"###### {category} Metrics Table (Aggregated for Task: {task_type_tab})")
#                                         filtered_df_dim_agg = task_df_agg[cols_to_show_present_agg_dim].copy()
#                                         new_cat_columns_agg = {}
#                                         for col_key_dim in filtered_df_dim_agg.columns:
#                                             if col_key_dim in metrics_in_category_task_agg: 
#                                                 indicator = get_metric_indicator(col_key_dim)
#                                                 col_title = get_metric_display_name(col_key_dim, include_placeholder_tag=False) 
#                                                 new_cat_columns_agg[col_key_dim] = f"{col_title} {indicator}".strip()
#                                             elif col_key_dim in ['model', 'num_samples']:
#                                                 new_cat_columns_agg[col_key_dim] = col_key_dim.replace('_', ' ').title()
#                                         filtered_df_dim_agg.rename(columns=new_cat_columns_agg, inplace=True)
#                                         display_dim_cols_agg = [new_cat_columns_agg.get(col,col) for col in cols_to_show_present_agg_dim if new_cat_columns_agg.get(col,col) in filtered_df_dim_agg.columns]
                                        
#                                         st.dataframe(
#                                             filtered_df_dim_agg[display_dim_cols_agg].style.pipe(apply_color_gradient, METRIC_INFO), 
#                                             use_container_width=True
#                                         )

#                                         st.markdown(f"###### {category} Charts (Aggregated for Task: {task_type_tab})")
#                                         plottable_metrics_agg = [m for m in metrics_in_category_task_agg if pd.api.types.is_numeric_dtype(task_df_agg[m])]
#                                         if not plottable_metrics_agg: st.info("No numeric metrics in this category for charting.")
#                                         else:
#                                             metric_display_options_agg = {f"{get_metric_display_name(m, False)} {get_metric_indicator(m)}".strip(): m for m in plottable_metrics_agg}
#                                             selected_metric_display_agg = st.selectbox(
#                                                 f"Metric for {task_type_tab} - {category}:", list(metric_display_options_agg.keys()),
#                                                 key=f"chart_sel_agg_{task_type_tab}_{category.replace(' ','_').replace('/','_')}_{i_task_tab}_{j_dim}" 
#                                             )
#                                             if selected_metric_display_agg:
#                                                 selected_metric_chart_agg = metric_display_options_agg[selected_metric_display_agg]
#                                                 metric_explanation_agg = METRIC_INFO.get(selected_metric_chart_agg, {}).get('explanation', "N/A")
#                                                 st.caption(f"**Definition ({get_metric_display_name(selected_metric_chart_agg, False)}):** {metric_explanation_agg}")
#                                                 try:
#                                                     fig_agg = px.bar(task_df_agg, x='model', y=selected_metric_chart_agg, title=f"{selected_metric_display_agg} Scores",
#                                                                 labels={'model': 'Model / Config', selected_metric_chart_agg: selected_metric_display_agg},
#                                                                 color='model', text_auto='.4f')
#                                                     fig_agg.update_layout(xaxis_title="Model / Config", yaxis_title=selected_metric_display_agg); fig_agg.update_traces(textposition='outside')
#                                                     st.plotly_chart(fig_agg, use_container_width=True)
#                                                 except Exception as e_chart: st.error(f"Chart error for {selected_metric_display_agg}: {e_chart}")
#             st.divider()
#             st.subheader("Download Aggregated Reports")
#             # ... (Download logic - unchanged)
#             if agg_df is not None and not agg_df.empty:
#                 col1_agg_dl, col2_agg_dl = st.columns(2)
#                 csv_data_agg = agg_df.to_csv(index=False, float_format="%.4f").encode('utf-8') 
#                 md_content_agg = f"# LLM Evaluation Aggregated Report ({datetime.datetime.now():%Y-%m-%d %H:%M})\n\n"
#                 agg_df_md_display_dl = agg_df.copy() 
#                 renamed_cols_md_dl = {}
#                 for col_md in agg_df_md_display_dl.columns:
#                     if col_md in METRIC_INFO:
#                         renamed_cols_md_dl[col_md] = get_metric_display_name(col_md, include_placeholder_tag=True) + \
#                                                      (f" {get_metric_indicator(col_md)}" if not is_placeholder_metric(col_md) else "")
#                     elif col_md in ['task_type', 'model', 'num_samples']:
#                          renamed_cols_md_dl[col_md] = col_md.replace('_', ' ').title()
#                 agg_df_md_display_dl.rename(columns=renamed_cols_md_dl, inplace=True)
#                 display_cols_for_md = [renamed_cols_md_dl.get(sc, sc) for sc in static_cols_display if renamed_cols_md_dl.get(sc,sc) in agg_df_md_display_dl.columns]
#                 sorted_metric_keys_for_md = sorted(
#                     [m for m in agg_df.columns if m in METRIC_INFO],
#                     key=lambda m: (is_placeholder_metric(m), METRIC_INFO[m]['category'], METRIC_INFO[m]['name'])
#                 )
#                 for m_key_md in sorted_metric_keys_for_md:
#                     renamed_m_key_md = renamed_cols_md_dl.get(m_key_md)
#                     if renamed_m_key_md and renamed_m_key_md in agg_df_md_display_dl.columns:
#                         display_cols_for_md.append(renamed_m_key_md)
#                 md_content_agg += agg_df_md_display_dl[display_cols_for_md].to_markdown(index=False, floatfmt=".4f")
#                 md_content_agg += "\n\n---\n_End of Aggregated Summary_"
#                 with col1_agg_dl: st.download_button("⬇️ CSV Aggregated Results", csv_data_agg, f"aggregated_eval_results_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv", key="dl_csv_agg")
#                 with col2_agg_dl: st.download_button("⬇️ MD Aggregated Summary", md_content_agg.encode('utf-8'), f"aggregated_eval_summary_{datetime.datetime.now():%Y%m%d_%H%M%S}.md", "text/markdown", key="dl_md_agg")
#         else: st.info("No aggregated results to display. Run an evaluation.")

#     with res_tab_ind:
#         # ... (Individual scores tab - largely same as genai_eval_tool_streamlit_app_may16_full) ...
#         st.subheader("📊 Individual Test Case Scores")
#         if st.session_state.individual_scores_df is not None and not st.session_state.individual_scores_df.empty:
#             ind_df_display = st.session_state.individual_scores_df.copy()
#             renamed_cols_ind_display = {}
#             original_metric_keys_in_ind_df = [col for col in ind_df_display.columns if col in METRIC_INFO]
#             for m_key in original_metric_keys_in_ind_df:
#                 renamed_cols_ind_display[m_key] = f"{get_metric_display_name(m_key, include_placeholder_tag=True)} {get_metric_indicator(m_key) if not is_placeholder_metric(m_key) else ''}".strip()
#             ind_df_display.rename(columns=renamed_cols_ind_display, inplace=True)

#             id_cols = ['id', 'task_type', 'model', 'test_description']
#             metric_cols_display_names = [renamed_cols_ind_display.get(m_key, m_key) for m_key in original_metric_keys_in_ind_df if renamed_cols_ind_display.get(m_key, m_key) in ind_df_display.columns]
#             interpretation_cols = ['Observations', 'Potential Actions', 'Metrics Not Computed or Not Applicable']
#             input_output_cols = ['question', 'ground_truth', 'answer', 'ref_facts', 'ref_key_points', 'contexts'] 
            
#             final_order_ind_display = []
#             for col_group in [id_cols, metric_cols_display_names, interpretation_cols, input_output_cols]:
#                 for col in col_group:
#                     if col in ind_df_display.columns and col not in final_order_ind_display: 
#                         final_order_ind_display.append(col)
#             remaining_other_cols_ind = sorted([col for col in ind_df_display.columns if col not in final_order_ind_display and not col.startswith('_st_')])
#             final_order_ind_display.extend(remaining_other_cols_ind)
#             final_order_ind_display = [col for col in final_order_ind_display if col in ind_df_display.columns] 

#             st.info("Displaying all scores and interpretations for each test case. Use column headers to sort. Download full table below.")
#             st.dataframe(ind_df_display[final_order_ind_display].style.pipe(apply_color_gradient, METRIC_INFO), use_container_width=True)
            
#             st.divider()
#             st.subheader("🔍 Detailed Interpretation for a Single Test Case")
#             if 'id' in st.session_state.individual_scores_df.columns:
#                 available_ids = st.session_state.individual_scores_df['id'].astype(str).unique().tolist()
#                 if not available_ids: st.warning("No test case IDs found.")
#                 else:
#                     options_for_selectbox = ["<Select a Test Case ID>"] + available_ids
#                     selected_id_for_interp = st.selectbox(
#                         "Select Test Case ID:", options=options_for_selectbox, index=0, key="individual_case_interp_selector"
#                     )
#                     if selected_id_for_interp and selected_id_for_interp != "<Select a Test Case ID>":
#                         selected_case_data = st.session_state.individual_scores_df[st.session_state.individual_scores_df['id'].astype(str) == selected_id_for_interp]
#                         if not selected_case_data.empty:
#                             case_to_show = selected_case_data.iloc[0] 
#                             st.markdown(f"**Test Case ID:** `{case_to_show.get('id', 'N/A')}`")
#                             st.markdown(f"**Model:** `{case_to_show.get('model', 'N/A')}`")
#                             st.markdown(f"**Task Type:** `{case_to_show.get('task_type', 'N/A')}`")
#                             if pd.notna(case_to_show.get('test_description')): st.markdown(f"**Description:** {case_to_show.get('test_description')}")
#                             st.markdown("---"); st.markdown("**Observations:**")
#                             obs_text = case_to_show.get('Observations', "_No specific observations generated._")
#                             st.markdown(obs_text if obs_text.strip() else "_No specific observations generated._")
#                             st.markdown("**Potential Actions:**")
#                             act_text = case_to_show.get('Potential Actions', "_No specific automated suggestions._")
#                             st.markdown(act_text if act_text.strip() else "_No specific automated suggestions._")
#                             st.markdown("**Metrics Not Computed or Not Applicable:**")
#                             na_text = case_to_show.get('Metrics Not Computed or Not Applicable', "_All relevant metrics computed or no notes._")
#                             st.markdown(na_text if na_text.strip() else "_All relevant metrics computed or no notes._")
#                             st.markdown("---")
#                             with st.expander("View Question, Ground Truth, and Answer"):
#                                 st.markdown(f"**Question:**\n```\n{case_to_show.get('question', '')}\n```")
#                                 st.markdown(f"**Ground Truth:**\n```\n{case_to_show.get('ground_truth', '')}\n```")
#                                 st.markdown(f"**LLM Answer:**\n```\n{case_to_show.get('answer', '')}\n```")
#                                 if pd.notna(case_to_show.get('contexts')): st.markdown(f"**Contexts (if provided):**\n```\n{case_to_show.get('contexts')}\n```")
#                         else: st.warning(f"Could not find data for ID: {selected_id_for_interp}")
#             else: st.info("Run evaluation to generate individual scores with IDs for detailed interpretation.")
#             st.divider(); st.subheader("Download Individual Scores Report (with Interpretations)")
#             # ... (Download logic for individual scores - unchanged)
#             csv_download_df_ind = st.session_state.individual_scores_df.copy() 
#             csv_id_cols = ['id', 'task_type', 'model', 'test_description']
#             csv_metric_cols = [m_key for m_key in METRIC_INFO.keys() if m_key in csv_download_df_ind.columns] 
#             csv_interp_cols = ['Observations', 'Potential Actions', 'Metrics Not Computed or Not Applicable']
#             csv_input_output_cols = ['question', 'ground_truth', 'answer', 'ref_facts', 'ref_key_points', 'contexts']
#             csv_final_order = []
#             for col_group in [csv_id_cols, csv_metric_cols, csv_interp_cols, csv_input_output_cols]:
#                 for col in col_group:
#                     if col in csv_download_df_ind.columns and col not in csv_final_order: csv_final_order.append(col)
#             csv_remaining_cols = sorted([col for col in csv_download_df_ind.columns if col not in csv_final_order and not col.startswith('_st_')])
#             csv_final_order.extend(csv_remaining_cols)
#             csv_final_order = [col for col in csv_final_order if col in csv_download_df_ind.columns] 
#             csv_data_ind = csv_download_df_ind[csv_final_order].to_csv(index=False, float_format="%.4f").encode('utf-8')
#             st.download_button("⬇️ CSV Individual Scores & Interpretations", csv_data_ind, f"individual_eval_scores_interpreted_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv", key="dl_csv_ind_interpreted")
#         else: st.info("No individual scores to display. Run an evaluation.")

# with tab_data_editor:
#     # ... (Data editor tab - use updated OPTIONAL_FIELDS_ADD_ROW_INFO for labels/placeholders)
#     st.header("Manage Evaluation Data")
#     st.markdown("Manually add new evaluation rows or edit existing data. Data loaded/generated via sidebar appears here.")
#     st.subheader("Add New Evaluation Row")
#     def update_input_mode(): st.session_state.add_row_input_mode = st.session_state.add_row_input_mode_selector_key
#     st.radio(
#         "Input Mode:", ("Easy (Required Fields Only)", "Custom (Select Additional Fields)"), 
#         key="add_row_input_mode_selector_key", horizontal=True,
#         index=("Easy (Required Fields Only)", "Custom (Select Additional Fields)").index(st.session_state.add_row_input_mode), 
#         on_change=update_input_mode 
#     )
#     with st.form("add_case_form", clear_on_submit=True):
#         col_form_id, col_form_task, col_form_model = st.columns(3)
#         with col_form_id: add_id_val = st.text_input("Test Case ID (Optional)", key="add_id_input", placeholder="e.g., rag_case_001")
#         with col_form_task: add_task_type_val = st.selectbox("Task Type*", list(get_supported_tasks()), key="add_task_type_select", index=None, placeholder="Select Task...")
#         with col_form_model: add_model_val = st.text_input("LLM/Model Config*", key="add_model_input", placeholder="e.g., MyModel_v1.2_temp0.7")
#         add_question_val = st.text_area("Question / Input Text*", key="add_question_input", placeholder="Input query, text to summarize/classify, or chatbot utterance.", height=100)
#         add_ground_truth_val = st.text_area("Ground Truth / Reference*", key="add_ground_truth_input", placeholder="Ideal answer, reference summary, correct label, or reference response.", height=100)
#         add_answer_val = st.text_area("LLM's Actual Answer / Prediction*", key="add_answer_input", placeholder="Actual output generated by the LLM.", height=100)
#         form_add_test_description = None; form_add_ref_facts = None; form_add_ref_key_points = None; form_add_contexts = None
#         if st.session_state.add_row_input_mode == "Custom (Select Additional Fields)":
#             st.markdown("---"); st.markdown("**Custom Fields (Optional):**")
#             form_add_test_description = st.text_input(OPTIONAL_FIELDS_ADD_ROW_INFO["test_description"]["label"], key="add_description_input_custom", placeholder=OPTIONAL_FIELDS_ADD_ROW_INFO["test_description"]["placeholder"])
#             st.caption(OPTIONAL_FIELDS_ADD_ROW_INFO["test_description"]["metric_info"])
#             form_add_ref_facts = st.text_input(OPTIONAL_FIELDS_ADD_ROW_INFO['ref_facts']['label'], key="add_ref_facts_input_custom", placeholder=OPTIONAL_FIELDS_ADD_ROW_INFO["ref_facts"]["placeholder"])
#             st.caption(OPTIONAL_FIELDS_ADD_ROW_INFO['ref_facts']['metric_info'])
#             form_add_ref_key_points = st.text_input(OPTIONAL_FIELDS_ADD_ROW_INFO['ref_key_points']['label'], key="add_ref_key_points_input_custom", placeholder=OPTIONAL_FIELDS_ADD_ROW_INFO["ref_key_points"]["placeholder"])
#             st.caption(OPTIONAL_FIELDS_ADD_ROW_INFO['ref_key_points']['metric_info'])
#             form_add_contexts = st.text_area("Contexts (Optional, for RAG)", key="add_contexts_input_custom", placeholder="Provide context snippets if relevant for RAG tasks.", height=75)
#         else: 
#             form_add_test_description = st.text_input(OPTIONAL_FIELDS_ADD_ROW_INFO["test_description"]["label"], key="add_description_input_easy", placeholder=OPTIONAL_FIELDS_ADD_ROW_INFO["test_description"]["placeholder"])
#             st.caption(OPTIONAL_FIELDS_ADD_ROW_INFO["test_description"]["metric_info"])
#         submitted_add_row = st.form_submit_button("➕ Add Evaluation Row to Editor")
#         if submitted_add_row:
#             # ... (Form submission logic - unchanged)
#             missing_fields = []
#             if not add_task_type_val: missing_fields.append("Task Type")
#             if not add_model_val.strip(): missing_fields.append("LLM/Model Config")
#             if not add_question_val.strip(): missing_fields.append("Question / Input Text")
#             if not add_ground_truth_val.strip(): missing_fields.append("Ground Truth / Reference")
#             if not add_answer_val.strip(): missing_fields.append("LLM's Actual Answer")
#             if missing_fields: st.error(f"Required fields (*) missing: {', '.join(missing_fields)}.")
#             else:
#                 current_df = st.session_state.edited_test_cases_df
#                 final_add_id = add_id_val.strip() if add_id_val and add_id_val.strip() else f"manual_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}"
#                 if not current_df.empty and 'id' in current_df.columns and final_add_id in current_df['id'].astype(str).values:
#                     st.error(f"ID '{final_add_id}' already exists. Please use a unique ID or leave blank for auto-generation.")
#                 else:
#                     new_row_dict = {
#                         'id': final_add_id, 'task_type': add_task_type_val, 
#                         'model': add_model_val.strip(), 'question': add_question_val.strip(),
#                         'ground_truth': add_ground_truth_val.strip(), 'answer': add_answer_val.strip(),
#                         'test_description': form_add_test_description.strip() if form_add_test_description else None,
#                         'ref_facts': form_add_ref_facts.strip() if form_add_ref_facts else None,
#                         'ref_key_points': form_add_ref_key_points.strip() if form_add_ref_key_points else None,
#                         'contexts': form_add_contexts.strip() if form_add_contexts else None,
#                     }
#                     new_row_dict_cleaned = {k: v for k, v in new_row_dict.items() if v is not None}
#                     new_row_df = pd.DataFrame([new_row_dict_cleaned])
#                     if st.session_state.edited_test_cases_df.empty: 
#                         st.session_state.edited_test_cases_df = new_row_df.fillna('')
#                     else:
#                         for col in new_row_df.columns:
#                             if col not in st.session_state.edited_test_cases_df.columns:
#                                 st.session_state.edited_test_cases_df[col] = np.nan 
#                         st.session_state.edited_test_cases_df = pd.concat(
#                             [st.session_state.edited_test_cases_df, new_row_df], ignore_index=True
#                         ).fillna('') 
#                     st.success(f"Row '{new_row_dict['id']}' added to editor.")
#     st.divider(); st.subheader("Data Editor")
#     # ... (Data editor display logic - unchanged)
#     if isinstance(st.session_state.edited_test_cases_df, pd.DataFrame) and not st.session_state.edited_test_cases_df.empty:
#         st.markdown("Edit data below. Changes are used when 'Run Evaluation' is clicked. Add/delete rows as needed.")
#         editor_col_order = ['id', 'task_type', 'model', 'test_description', 
#                             'question', 'ground_truth', 'answer', 
#                             'ref_facts', 'ref_key_points', 'contexts'] 
#         current_cols_in_df = st.session_state.edited_test_cases_df.columns.tolist()
#         for col in editor_col_order:
#             if col not in current_cols_in_df: st.session_state.edited_test_cases_df[col] = '' 
#         final_editor_cols = [col for col in editor_col_order if col in st.session_state.edited_test_cases_df.columns]
#         remaining_cols_for_editor = sorted([col for col in st.session_state.edited_test_cases_df.columns if col not in final_editor_cols])
#         final_editor_cols.extend(remaining_cols_for_editor)
#         df_for_editor_display = st.session_state.edited_test_cases_df[final_editor_cols].copy().fillna('')
#         edited_df_from_editor = st.data_editor(
#             df_for_editor_display, num_rows="dynamic", use_container_width=True, key="data_editor_main"
#         )
#         st.session_state.edited_test_cases_df = edited_df_from_editor.copy() 
#         if not st.session_state.edited_test_cases_df.empty:
#              csv_edited_data = st.session_state.edited_test_cases_df.fillna('').to_csv(index=False).encode('utf-8')
#              st.download_button("⬇️ Download Edited Data (CSV)", csv_edited_data, f"edited_eval_cases_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv", key="dl_edited_data_csv")
#     else: st.info("No data loaded. Use sidebar to load/generate or add rows using the form.")

# with tab_format_guide:
#     # ... (Data format guide - use updated descriptions for ref_facts/ref_key_points)
#     st.header("Input Data Format Guide (Flat Format)")
#     st.markdown("Framework expects input (JSON, CSV, Excel) in a **flat format**. Each row is one evaluation instance.")
#     st.markdown("**Required Columns:** `task_type`, `model`, `question`, `ground_truth`, `answer`.")
#     st.markdown("**Optional Columns for Specific Metrics:**")
#     st.markdown("- `ref_facts`: Comma-separated *exact factual statements* the answer should contain (e.g., `The capital is Paris,Currency is Euro`). Used by **Fact Presence** score. Case-insensitive check.")
#     st.markdown("- `ref_key_points`: Comma-separated *key topics/themes* the answer should cover (e.g., `product benefits,installation,support`). Used by **Key Point Coverage** score. Case-insensitive check.")
#     st.markdown("**Other Optional Columns:**")
#     st.markdown("- `id`: Unique row identifier (recommended).")
#     st.markdown("- `test_description`: Brief description of the test case.")
#     st.markdown("- `contexts`: (Optional for RAG) Context snippets for RAG tasks.")
#     st.subheader("Example Rows (Conceptual):")
#     example_data_guide = [
#         {'id': 'rag_001', 'task_type': 'rag_faq', 'model': 'ModelAlpha', 'test_description': 'Capital of France', 'question': 'Info on Paris?', 'contexts': 'Paris is capital...', 'ground_truth': 'Paris is the capital of France and a major global city.', 'answer': 'Paris is France\'s capital.', 'ref_facts': 'Paris is capital of France', 'ref_key_points': 'Capital city,Global city status'},
#         {'id': 'sum_001', 'task_type': 'summarization', 'model': 'ModelBeta', 'test_description': 'AI Summary', 'question': 'Summarize AI impact.', 'contexts': '', 'ground_truth': 'AI has transformed industries...', 'answer': 'AI changed many fields.', 'ref_facts': '', 'ref_key_points': 'Industry transformation,Societal impact'}
#     ]
#     st.dataframe(pd.DataFrame(example_data_guide).fillna(''))


# with tab_metrics_tutorial:
#     # ... (Metrics tutorial - largely same as genai_eval_tool_streamlit_app_may16_full, ensures updated explanations are used)
#     st.header("Metrics Tutorial & Explanations")
#     st.markdown("Understand the metrics used. Metrics are grouped by evaluation dimension. Metrics requiring specific optional inputs (like `ref_facts`) will result in a `NaN` score if those inputs are not provided in your data. Placeholder metrics are not fully implemented and will also return `NaN`.")
#     for category_tut in CATEGORY_ORDER: 
#         metrics_in_this_category_tut = METRICS_BY_CATEGORY.get(category_tut, [])
#         if not metrics_in_this_category_tut: continue
#         with st.expander(f"**{category_tut}**", expanded=(category_tut == CAT_TRUST)): 
#             st.markdown(f"*{DIMENSION_DESCRIPTIONS.get(category_tut, '')}*"); st.markdown("---")
#             if not metrics_in_this_category_tut: st.markdown("_No metrics currently assigned to this dimension._")
#             else:
#                 for metric_key_tut in metrics_in_this_category_tut:
#                     info_tut = METRIC_INFO.get(metric_key_tut)
#                     if info_tut:
#                         indicator_tut = get_metric_indicator(metric_key_tut) if not is_placeholder_metric(metric_key_tut) else ""
#                         display_name_tut = get_metric_display_name(metric_key_tut, include_placeholder_tag=True)
#                         st.markdown(f"##### {display_name_tut} (`{metric_key_tut}`) {indicator_tut}")
#                         explanation_text_tut = info_tut['explanation']
#                         if is_placeholder_metric(metric_key_tut): explanation_text_tut = f"**[PLACEHOLDER]** {explanation_text_tut}"
#                         st.markdown(f"**Use Case & Interpretation:** {explanation_text_tut}")
#                         relevant_tasks_tut = [task_name for task_name in get_supported_tasks() if metric_key_tut in get_metrics_for_task(task_name)]
#                         if relevant_tasks_tut: st.markdown(f"**Commonly Used For Tasks:** `{'`, `'.join(relevant_tasks_tut)}`")
#                         else: st.markdown("**Commonly Used For Tasks:** (General or not task-specific).")
#                         input_field_data_key_tut = info_tut.get("input_field_data_key")
#                         if input_field_data_key_tut: st.markdown(f"**Relies on Input Data Field:** `{input_field_data_key_tut}` (Score NaN if missing).")
#                         if metric_key_tut == SEMANTIC_SIMILARITY_SCORE: st.markdown(f"**Note:** Requires `sentence-transformers`. May download model on first run.")
#                         st.markdown("---")







# streamlit_app.py (Main Orchestrator)
import streamlit as st
import pandas as pd
import numpy as np
import traceback # For error logging
from pathlib import Path
import sys

# --- Add project root to sys.path for local development ---
# This structure assumes 'src' and 'streamlit_app.py' are at the project root,
# and modules are in 'src/' or 'src/ui_components/'
# Adjust if your structure is different.
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root)) # Adds GenAI_Evaluation_Tool-main
sys.path.insert(0, str(project_root / "src")) # Adds GenAI_Evaluation_Tool-main/src

# --- Import Core Logic Modules ---
from src.evaluator import evaluate_model_responses
from src.tasks.task_registry import get_supported_tasks, RAG_FAQ, SUMMARIZATION, CLASSIFICATION, CHATBOT # For run eval button
# --- Import Configuration & UI Helpers ---
from src.app_config import METRIC_INFO, DIMENSION_DESCRIPTIONS, CATEGORY_ORDER, OPTIONAL_FIELDS_ADD_ROW_INFO
from src.ui_helpers import initialize_session_state, clear_app_state, unflatten_df_to_test_cases
# --- Import UI Component Rendering Functions ---
from src.ui_components.sidebar_view import render_sidebar
from src.ui_components.data_management_view import render_data_editor_tab, render_data_format_guide_tab
from src.ui_components.results_view import render_individual_scores_sub_tab, render_aggregated_results_sub_tab
from src.ui_components.tutorial_view import render_metrics_tutorial_tab
# --- Import Interpretation Engine ---
import src.interpretation_engine as interpretation_engine # Keep it namespaced

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="LLM Evaluation Framework")
st.title("📊 LLM Evaluation Framework")
st.markdown("Evaluate LLM performance using pre-generated responses. This tool supports aggregated summaries and individual test case scores, including lexical and semantic similarity.")

# --- Initialize Session State ---
initialize_session_state() # Ensures all necessary keys are in st.session_state

# --- Render Sidebar and Handle Data Loading ---
# render_sidebar() will update st.session_state.edited_test_cases_df and st.session_state.data_source_info
render_sidebar()


# --- Main Application Tabs ---
tab_eval, tab_data_editor, tab_format_guide, tab_metrics_tutorial = st.tabs([
    "📊 Evaluation & Results", 
    "📝 View/Edit/Add Data", 
    "📄 Data Format Guide", 
    "📖 Metrics Tutorial"
])

# --- Evaluation & Results Tab ---
with tab_eval:
    st.header("Run Evaluation and View Results")
    run_button_disabled = not isinstance(st.session_state.edited_test_cases_df, pd.DataFrame) or st.session_state.edited_test_cases_df.empty
    
    st.info(
        "ℹ️ **Semantic Similarity Note:** If this is your first time running with Semantic Similarity, "
        "the 'sentence-transformers' library may need to download model files (e.g., 'all-MiniLM-L6-v2'), "
        "which requires an internet connection. This download happens once per model. "
        "Subsequent runs will use the cached model. For offline use, pre-download the model (see documentation/tool output)."
    )

    if st.button("🚀 Run Evaluation on Data in Editor", disabled=run_button_disabled, key="run_eval_main_button_app", help="Evaluates the current data shown in the 'View/Edit/Add Data' tab."):
        if isinstance(st.session_state.edited_test_cases_df, pd.DataFrame) and not st.session_state.edited_test_cases_df.empty:
            st.session_state.aggregated_results_df = None
            st.session_state.individual_scores_df = None
            st.session_state.metrics_for_agg_display = [] # Reset this
            
            with st.spinner("⏳ Evaluating... This may take a moment, especially if downloading models for semantic similarity."):
                try:
                    df_to_process = st.session_state.edited_test_cases_df.replace('', np.nan)
                    test_cases_to_evaluate = unflatten_df_to_test_cases(df_to_process)
                    
                    if not test_cases_to_evaluate:
                        raise ValueError("Data in editor is empty or could not be processed into valid test cases. Ensure required fields (task_type, model, question, ground_truth, answer) are filled.")
                    
                    individual_df_raw, aggregated_df = evaluate_model_responses(test_cases_to_evaluate)
                    
                    if individual_df_raw is not None and not individual_df_raw.empty:
                        # Use the interpretation engine
                        interpretations_output = individual_df_raw.apply(
                            lambda row: interpretation_engine.generate_single_case_interpretation(row, row.get('task_type')), axis=1
                        )
                        individual_df_raw['Observations'] = interpretations_output.apply(lambda x: x[0])
                        individual_df_raw['Potential Actions'] = interpretations_output.apply(lambda x: x[1])
                        individual_df_raw['Metrics Not Computed or Not Applicable'] = interpretations_output.apply(lambda x: x[2])
                    
                    st.session_state.individual_scores_df = individual_df_raw
                    st.session_state.aggregated_results_df = aggregated_df

                    if aggregated_df is not None and not aggregated_df.empty:
                        metrics_present_in_agg = [col for col in aggregated_df.columns if col in METRIC_INFO]
                        metrics_to_show_agg = []
                        for metric_col in metrics_present_in_agg:
                            # Use ui_helpers.is_placeholder_metric
                            from src.ui_helpers import is_placeholder_metric as ipm_check # Local import for clarity
                            if pd.api.types.is_numeric_dtype(aggregated_df[metric_col]):
                                if not (aggregated_df[metric_col].isna() | (aggregated_df[metric_col].abs() < 1e-9)).all() \
                                   and not ipm_check(metric_col): # Check if not placeholder
                                    metrics_to_show_agg.append(metric_col)
                        st.session_state.metrics_for_agg_display = metrics_to_show_agg
                        st.success("✅ Evaluation complete! View results below.")
                    elif individual_df_raw is not None and not individual_df_raw.empty:
                         st.warning("⚠️ Evaluation produced individual scores, but aggregated results are empty or only contain placeholders/NaNs. This might happen if all metric scores were NaN or zero across all data, or if only placeholder metrics were computed.")
                    else:
                        st.warning("⚠️ Evaluation finished, but no results (neither individual nor aggregated) were produced. Check input data and console logs.")
                except Exception as e:
                     st.error(f"Evaluation error: {e}")
                     st.text_area("Traceback", traceback.format_exc(), height=150) # Show traceback in UI
        else:
            st.warning("No data in editor. Load or generate data first.")

    st.divider()
    st.header("Evaluation Results")
    res_tab_ind, res_tab_agg = st.tabs(["📄 Individual Scores", "📈 Aggregated Results"])

    with res_tab_ind:
        render_individual_scores_sub_tab() # Call from ui_components.results_view

    with res_tab_agg:
        render_aggregated_results_sub_tab(interpretation_engine) # Call from ui_components.results_view

# --- Data Editor Tab ---
with tab_data_editor:
    render_data_editor_tab() # Call from ui_components.data_management_view

# --- Data Format Guide Tab ---
with tab_format_guide:
    render_data_format_guide_tab() # Call from ui_components.data_management_view

# --- Metrics Tutorial Tab ---
with tab_metrics_tutorial:
    render_metrics_tutorial_tab() # Call from ui_components.tutorial_view

