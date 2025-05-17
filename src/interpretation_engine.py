# src/interpretation_engine.py
"""
Handles the logic for generating textual interpretations of metric scores
for both individual test cases and aggregated results.
"""
import numpy as np
import pandas as pd

# Import config and helpers
from .app_config import METRIC_INFO, SEMANTIC_SIMILARITY_SCORE, CLASSIFICATION
from .ui_helpers import get_metric_display_name, is_placeholder_metric

def generate_single_case_interpretation(case_row, task_type):
    """
    Generates observations, suggestions, and notes on non-applicable metrics
    for a single test case row.
    (Code for this function remains the same as previously provided in
     "genai_eval_tool_interpretation_engine_may17")
    """
    observations = []
    suggestions = []
    not_applicable_metrics = []

    overall_assessment_flags = {
        "fluency": None, "semantic": None, "factuality": None, "completeness": None,
        "classification": None, "conciseness": None, "safety": "ok", "privacy": "ok"
    }

    def was_input_provided(metric_key, case_data):
        metric_detail = METRIC_INFO.get(metric_key)
        if metric_detail and "input_field_data_key" in metric_detail:
            input_key = metric_detail["input_field_data_key"]
            # For individual case, case_data is the row itself (a Series)
            return input_key in case_data.index and pd.notna(case_data[input_key]) and str(case_data[input_key]).strip()
        return True

    # --- Fluency & Lexical Similarity ---
    fluency_metric_keys = ['bleu', 'rouge_1', 'rouge_2', 'rouge_l', 'meteor']
    valid_fluency_scores = []
    for key in fluency_metric_keys:
        if key in case_row.index and pd.notna(case_row[key]) and not is_placeholder_metric(key):
            valid_fluency_scores.append(case_row[key])
        elif key in case_row.index and pd.isna(case_row[key]) and not is_placeholder_metric(key):
            not_applicable_metrics.append(f"{get_metric_display_name(key, False)}: Score is NaN.")
    if valid_fluency_scores:
        avg_fluency = np.mean(valid_fluency_scores)
        if avg_fluency >= 0.6: observations.append(f"✅ Fluency & Lexical Sim.: Strong (Avg. Score: {avg_fluency:.2f})."); overall_assessment_flags["fluency"] = "good"
        elif avg_fluency >= 0.3: observations.append(f"⚠️ Fluency & Lexical Sim.: Moderate (Avg. Score: {avg_fluency:.2f})."); overall_assessment_flags["fluency"] = "moderate"
        else: observations.append(f"❌ Fluency & Lexical Sim.: Low (Avg. Score: {avg_fluency:.2f})."); overall_assessment_flags["fluency"] = "poor"; suggestions.append("Lexical similarity is low. Review for grammar, coherence. Compare with Semantic Similarity.")

    # --- Semantic Understanding ---
    metric_key_sem_sim = SEMANTIC_SIMILARITY_SCORE
    if metric_key_sem_sim in case_row.index: # Check if column exists
        score_sem_sim = case_row[metric_key_sem_sim]
        if is_placeholder_metric(metric_key_sem_sim): not_applicable_metrics.append(f"{get_metric_display_name(metric_key_sem_sim, False)}: Placeholder unexpectedly.")
        elif pd.notna(score_sem_sim):
            if score_sem_sim >= 0.75: observations.append(f"✅ Semantic Similarity: Strong ({score_sem_sim:.2f}). Meaning highly aligned."); overall_assessment_flags["semantic"] = "good"
            elif score_sem_sim >= 0.5: observations.append(f"ℹ️ Semantic Similarity: Moderate ({score_sem_sim:.2f}). Meaning somewhat aligned."); overall_assessment_flags["semantic"] = "moderate"
            elif score_sem_sim >= 0.25: observations.append(f"⚠️ Semantic Similarity: Fair ({score_sem_sim:.2f}). Some semantic overlap but may miss key aspects."); overall_assessment_flags["semantic"] = "fair"
            else: observations.append(f"❌ Semantic Similarity: Low ({score_sem_sim:.2f}). Meaning diverges significantly."); overall_assessment_flags["semantic"] = "poor"; suggestions.append("Semantic similarity is low. Model may misunderstand query or diverge factually.")
        elif pd.isna(score_sem_sim):
            sem_sim_metric_info = METRIC_INFO.get(metric_key_sem_sim, {})
            explanation = sem_sim_metric_info.get("explanation", "").lower()
            if "sentence-transformers library" in explanation or "failed to load" in explanation:
                 not_applicable_metrics.append(f"{get_metric_display_name(metric_key_sem_sim, False)}: Not computed (Sentence Transformers library/model issue).")
            else: not_applicable_metrics.append(f"{get_metric_display_name(metric_key_sem_sim, False)}: Score is NaN.")

    # --- Trust & Factuality (Fact Presence) ---
    metric_key_fp = 'fact_presence_score'
    if metric_key_fp in case_row.index:
        score_fp = case_row[metric_key_fp]
        # For individual case, was_input_provided checks the 'ref_facts' in the original test case data,
        # which should be part of 'case_row' if it was used to generate the score.
        if not was_input_provided(metric_key_fp, case_row): not_applicable_metrics.append(f"{get_metric_display_name(metric_key_fp, False)}: Not applicable (missing `ref_facts`).")
        elif pd.notna(score_fp):
            if score_fp >= 0.7: observations.append(f"✅ Fact Presence: Good ({score_fp:.2f}). Most specified facts included."); overall_assessment_flags["factuality"] = "good"
            elif score_fp >= 0.4: observations.append(f"⚠️ Fact Presence: Moderate ({score_fp:.2f}). Some facts missing/altered."); overall_assessment_flags["factuality"] = "moderate"
            else: observations.append(f"❌ Fact Presence: Low ({score_fp:.2f}). Significant facts missing."); overall_assessment_flags["factuality"] = "poor"; suggestions.append("Verify missing/inaccurate critical facts from `ref_facts`.")
        elif pd.isna(score_fp):
            # If score is NaN but input was expected to be there (implicit if was_input_provided was true or not applicable for this metric type)
             not_applicable_metrics.append(f"{get_metric_display_name(metric_key_fp, False)}: Score is NaN (Input `ref_facts` might have been provided; review calculation).")


    # --- Completeness & Coverage (Key Point Coverage) ---
    metric_key_cc = 'completeness_score'
    if metric_key_cc in case_row.index:
        score_cc = case_row[metric_key_cc]
        if not was_input_provided(metric_key_cc, case_row): not_applicable_metrics.append(f"{get_metric_display_name(metric_key_cc, False)}: Not applicable (missing `ref_key_points`).")
        elif pd.notna(score_cc):
            if score_cc >= 0.7: observations.append(f"✅ Key Point Coverage: Good ({score_cc:.2f}). Most key points covered."); overall_assessment_flags["completeness"] = "good"
            elif score_cc >= 0.4: observations.append(f"⚠️ Key Point Coverage: Moderate ({score_cc:.2f}). Some key points unaddressed."); overall_assessment_flags["completeness"] = "moderate"
            else: observations.append(f"❌ Key Point Coverage: Low ({score_cc:.2f}). Significant key points missing."); overall_assessment_flags["completeness"] = "poor"; suggestions.append("Verify if essential topics from `ref_key_points` were addressed.")
        elif pd.isna(score_cc):
             not_applicable_metrics.append(f"{get_metric_display_name(metric_key_cc, False)}: Score is NaN (Input `ref_key_points` might have been provided; review calculation).")

    # --- Classification Accuracy ---
    if task_type == CLASSIFICATION:
        if 'accuracy' in case_row.index:
            accuracy = case_row['accuracy']
            if pd.notna(accuracy):
                if accuracy < 1.0: observations.append(f"❌ Classification: Incorrect (Accuracy: {accuracy:.2f})."); overall_assessment_flags["classification"] = "incorrect"; suggestions.append("Analyze misclassification. Review input/ground truth.")
                else: observations.append(f"✅ Classification: Correct (Accuracy: {accuracy:.2f})."); overall_assessment_flags["classification"] = "correct"
            elif pd.isna(accuracy): not_applicable_metrics.append(f"{get_metric_display_name('accuracy', False)}: Score is NaN.")

    # --- Conciseness ---
    if 'length_ratio' in case_row.index:
        length_ratio = case_row['length_ratio']
        if pd.notna(length_ratio):
            if length_ratio < 0.5: observations.append(f"⚠️ Conciseness: Response significantly shorter (ratio: {length_ratio:.2f})."); overall_assessment_flags["conciseness"] = "too_short"; suggestions.append("Check if response is overly brief/truncated.")
            elif length_ratio < 0.8: observations.append(f"ℹ️ Conciseness: Response noticeably shorter (ratio: {length_ratio:.2f})."); overall_assessment_flags["conciseness"] = "short"
            elif length_ratio <= 1.25: observations.append(f"✅ Conciseness: Response length comparable to reference (ratio: {length_ratio:.2f})."); overall_assessment_flags["conciseness"] = "good"
            elif length_ratio <= 1.75: observations.append(f"ℹ️ Conciseness: Response noticeably longer (ratio: {length_ratio:.2f})."); overall_assessment_flags["conciseness"] = "long"
            else: observations.append(f"⚠️ Conciseness: Response significantly longer (ratio: {length_ratio:.2f})."); overall_assessment_flags["conciseness"] = "too_long"; suggestions.append("Check if response is too verbose/irrelevant.")
        elif pd.isna(length_ratio): not_applicable_metrics.append(f"{get_metric_display_name('length_ratio', False)}: Score is NaN.")

    # --- Safety & Privacy (Only show if issue detected) ---
    if 'safety_keyword_score' in case_row.index:
        safety_score = case_row['safety_keyword_score']
        if pd.notna(safety_score):
            if safety_score < 1.0:
                observations.append("🚨 Safety: Potential safety keyword detected (Basic Check).")
                overall_assessment_flags["safety"] = "issue"
                suggestions.append("MANUAL REVIEW REQUIRED for safety. Identify problematic content; refine safety filters/prompts.")
        elif pd.isna(safety_score):
            not_applicable_metrics.append(f"{get_metric_display_name('safety_keyword_score', False)}: Score is NaN.")

    if 'pii_detection_score' in case_row.index:
        pii_score = case_row['pii_detection_score']
        if pd.notna(pii_score):
            if pii_score < 1.0:
                observations.append("🚨 Privacy: Potential PII pattern detected (Basic Regex Check).")
                overall_assessment_flags["privacy"] = "issue"
                suggestions.append("MANUAL REVIEW REQUIRED for PII. Ensure sensitive data is not exposed; enhance PII detection/scrubbing.")
        elif pd.isna(pii_score):
            not_applicable_metrics.append(f"{get_metric_display_name('pii_detection_score', False)}: Score is NaN.")

    # Handle placeholder metrics
    placeholder_keys_to_check = ["professional_tone_score", "refusal_quality_score", "nli_entailment_score", "llm_judge_factuality"]
    for pk in placeholder_keys_to_check:
        if pk in case_row.index:
            metric_display_name_pk = get_metric_display_name(pk, False)
            if is_placeholder_metric(pk):
                metric_explanation = METRIC_INFO.get(pk, {}).get('explanation', 'Full implementation pending.')
                if pd.isna(case_row[pk]):
                    not_applicable_metrics.append(f"{metric_display_name_pk}: Placeholder - Not implemented. ({metric_explanation})")
                else:
                    not_applicable_metrics.append(f"{metric_display_name_pk}: Placeholder received unexpected score ({case_row[pk]:.2f}).")

    # --- Final Summary for Suggestions ---
    final_suggestions_summary_parts = []
    issues_found_text = []
    positives_found_text = []

    if overall_assessment_flags["fluency"] == "poor": issues_found_text.append("low fluency/lexical similarity")
    elif overall_assessment_flags["fluency"] == "good": positives_found_text.append("good fluency/lexical similarity")
    if overall_assessment_flags["semantic"] == "poor": issues_found_text.append("low semantic similarity")
    elif overall_assessment_flags["semantic"] == "good": positives_found_text.append("good semantic similarity")
    if overall_assessment_flags["factuality"] == "poor": issues_found_text.append("low fact presence")
    elif overall_assessment_flags["factuality"] == "good": positives_found_text.append("good fact presence")
    if overall_assessment_flags["completeness"] == "poor": issues_found_text.append("low key point coverage")
    elif overall_assessment_flags["completeness"] == "good": positives_found_text.append("good key point coverage")
    if overall_assessment_flags["classification"] == "incorrect": issues_found_text.append("incorrect classification")
    elif overall_assessment_flags["classification"] == "correct": positives_found_text.append("correct classification")
    if overall_assessment_flags["conciseness"] == "too_short": issues_found_text.append("response too short")
    elif overall_assessment_flags["conciseness"] == "too_long": issues_found_text.append("response too long")
    elif overall_assessment_flags["conciseness"] == "good": positives_found_text.append("appropriate length")
    if overall_assessment_flags["safety"] == "issue": issues_found_text.append("potential safety concerns")
    if overall_assessment_flags["privacy"] == "issue": issues_found_text.append("potential PII exposure")

    if issues_found_text:
        summary_statement = f"Overall, potential concerns regarding: {', '.join(issues_found_text)}. "
        if positives_found_text: summary_statement += f"However, performed well in: {', '.join(positives_found_text)}. "
        summary_statement += "Consider specific suggestions."
        final_suggestions_summary_parts.append(summary_statement)
    elif positives_found_text:
        final_suggestions_summary_parts.append(f"Overall, performed well regarding: {', '.join(positives_found_text)}. No major issues flagged by automated metrics.")
    else:
        if not observations and not_applicable_metrics: final_suggestions_summary_parts.append("Most metrics not applicable/computed. Manual review needed for specific evaluations.")
        elif not observations : final_suggestions_summary_parts.append("No specific issues/strengths flagged by automated metrics. Manual review recommended for nuanced aspects.")

    final_suggestions_text = "\n".join(f"- {s_item}" for s_item in final_suggestions_summary_parts + suggestions if s_item)
    if not final_suggestions_text.strip(): final_suggestions_text = "- Review case manually for unmeasured criteria."

    if not observations and not_applicable_metrics: observations.append("No specific metric observations (scores might be NaN, metrics not applicable, or placeholders). See 'Metrics Not Computed or Not Applicable'.")
    elif not observations: observations.append("No specific metric observations generated. Scores might be within acceptable ranges or metrics not applicable for this task.")

    return "\n".join(f"- {o}" for o in observations), final_suggestions_text, "\n".join(f"- {na}" for na in not_applicable_metrics if na)


def generate_aggregated_interpretations(model_scores_row, task_type):
    """
    Generates observations, suggestions, and notes on non-applicable metrics
    for a row of aggregated scores (typically mean scores for a model/task).
    Args:
        model_scores_row (pd.Series): A row from the aggregated_scores_df.
        task_type (str): The task type for this aggregated group.
    Returns:
        tuple: (observations_str, suggestions_str, not_applicable_str)
    """
    observations = []
    suggestions = []
    not_applicable_agg = [] # For metrics that are NaN at the aggregated level

    # --- Fluency & Lexical Similarity (Aggregated) ---
    fluency_keys = ['bleu', 'rouge_l', 'meteor'] # Key metrics for overall fluency
    agg_fluency_scores = [model_scores_row.get(k) for k in fluency_keys if k in model_scores_row and pd.notna(model_scores_row.get(k)) and not is_placeholder_metric(k)]
    if agg_fluency_scores:
        avg_fluency_agg = np.mean(agg_fluency_scores)
        if avg_fluency_agg > 0.5: observations.append(f"✅ Generally good fluency & lexical similarity (Avg. relevant score: {avg_fluency_agg:.2f}).")
        elif avg_fluency_agg > 0.2: observations.append(f"⚠️ Moderate fluency/lexical similarity on average (Avg. score: {avg_fluency_agg:.2f}). Responses may often differ lexically from references.")
        else: observations.append(f"❌ Low fluency/lexical similarity on average (Avg. score: {avg_fluency_agg:.2f}). Responses might frequently be quite different or have linguistic issues."); suggestions.append("Review individual low-scoring cases for clarity and grammar. Compare with aggregated semantic similarity.")
    else: # Check if any fluency keys were expected but all were NaN
        for fk in fluency_keys:
            if fk in model_scores_row and pd.isna(model_scores_row[fk]) and not is_placeholder_metric(fk):
                not_applicable_agg.append(f"{get_metric_display_name(fk, False)}: Aggregated score is NaN (likely NaN for all individual cases or not computed).")


    # --- Semantic Understanding (Aggregated) ---
    agg_semantic_sim = model_scores_row.get(SEMANTIC_SIMILARITY_SCORE)
    if pd.notna(agg_semantic_sim) and not is_placeholder_metric(SEMANTIC_SIMILARITY_SCORE):
        if agg_semantic_sim > 0.7: observations.append(f"✅ Good average semantic similarity to references ({agg_semantic_sim:.2f}). Meaning is generally well-aligned.")
        elif agg_semantic_sim > 0.4: observations.append(f"ℹ️ Moderate average semantic similarity ({agg_semantic_sim:.2f}). Meaning is somewhat aligned on average.")
        else: observations.append(f"⚠️ Low average semantic similarity ({agg_semantic_sim:.2f}). Meaning may differ significantly on average."); suggestions.append("Low average semantic similarity can indicate systemic misunderstanding of queries or factual divergence.")
    elif SEMANTIC_SIMILARITY_SCORE in model_scores_row and pd.isna(agg_semantic_sim) and not is_placeholder_metric(SEMANTIC_SIMILARITY_SCORE):
        not_applicable_agg.append(f"{get_metric_display_name(SEMANTIC_SIMILARITY_SCORE, False)}: Aggregated score is NaN (check individual cases or model loading issues).")


    # --- Trust & Factuality (Fact Presence - Aggregated) ---
    agg_fact_presence = model_scores_row.get('fact_presence_score')
    if pd.notna(agg_fact_presence) and not is_placeholder_metric('fact_presence_score'):
        if agg_fact_presence > 0.7: observations.append(f"✅ Good average inclusion of specified facts ({agg_fact_presence:.2f}).")
        elif agg_fact_presence > 0.4: observations.append(f"⚠️ Moderate average inclusion of facts ({agg_fact_presence:.2f}). Some key information might often be missing.")
        else: observations.append(f"❌ Low average inclusion of specified facts ({agg_fact_presence:.2f})."); suggestions.append("Review if `ref_facts` were consistently provided and accurate for this group. Model may need better prompting for fact extraction if inputs were correct.")
    elif 'fact_presence_score' in model_scores_row and pd.isna(agg_fact_presence) and not is_placeholder_metric('fact_presence_score'):
        not_applicable_agg.append(f"{get_metric_display_name('fact_presence_score', False)}: Aggregated score is NaN (likely `ref_facts` were consistently missing or all individual scores were NaN).")

    # --- Completeness & Coverage (Key Point Coverage - Aggregated) ---
    agg_completeness = model_scores_row.get('completeness_score')
    if pd.notna(agg_completeness) and not is_placeholder_metric('completeness_score'):
        if agg_completeness > 0.7: observations.append(f"✅ Good average coverage of key points ({agg_completeness:.2f}).")
        elif agg_completeness > 0.4: observations.append(f"⚠️ Moderate average coverage of key points ({agg_completeness:.2f}). May often not address all aspects.")
        else: observations.append(f"❌ Low average coverage of key points ({agg_completeness:.2f})."); suggestions.append("Review if `ref_key_points` were consistently provided and well-defined. Model might need prompting for better comprehensiveness.")
    elif 'completeness_score' in model_scores_row and pd.isna(agg_completeness) and not is_placeholder_metric('completeness_score'):
        not_applicable_agg.append(f"{get_metric_display_name('completeness_score', False)}: Aggregated score is NaN (likely `ref_key_points` were consistently missing or all individual scores were NaN).")

    # --- Classification (Aggregated F1-Score or Accuracy) ---
    if task_type == CLASSIFICATION:
        agg_f1 = model_scores_row.get('f1_score')
        agg_acc = model_scores_row.get('accuracy')
        # Prioritize F1 for aggregated classification interpretation
        if pd.notna(agg_f1) and not is_placeholder_metric('f1_score'):
            if agg_f1 > 0.75: observations.append(f"✅ Good overall classification performance (Avg. F1: {agg_f1:.2f}).")
            elif agg_f1 > 0.5: observations.append(f"⚠️ Moderate overall classification performance (Avg. F1: {agg_f1:.2f}).")
            else: observations.append(f"❌ Low overall classification performance (Avg. F1: {agg_f1:.2f})."); suggestions.append("Review misclassified examples across the dataset. Consider if model needs more/better training data or feature engineering.")
        elif pd.notna(agg_acc) and not is_placeholder_metric('accuracy'): # Fallback to accuracy
             if agg_acc > 0.75: observations.append(f"✅ Good overall classification accuracy (Avg. Accuracy: {agg_acc:.2f}).")
             else: interpretations.append(f"⚠️ Overall classification accuracy is {agg_acc:.2f}. Check F1/Precision/Recall if available."); suggestions.append("Review misclassified examples if overall accuracy is low.")
        elif ('f1_score' in model_scores_row and pd.isna(agg_f1) and not is_placeholder_metric('f1_score')) or \
             ('accuracy' in model_scores_row and pd.isna(agg_acc) and not is_placeholder_metric('accuracy')):
            not_applicable_agg.append("Classification Metrics (F1/Accuracy): Aggregated scores are NaN.")


    # --- Conciseness (Aggregated Length Ratio) ---
    agg_length_ratio = model_scores_row.get('length_ratio')
    if pd.notna(agg_length_ratio) and not is_placeholder_metric('length_ratio'):
        if 0.75 <= agg_length_ratio <= 1.25: observations.append(f"✅ Good average response length relative to reference (Avg. Ratio: {agg_length_ratio:.2f}).")
        elif agg_length_ratio < 0.5: observations.append(f"⚠️ Responses may be too short on average (Avg. Ratio: {agg_length_ratio:.2f})."); suggestions.append("Check if model is consistently truncating answers or being overly brief.")
        elif agg_length_ratio > 1.75: observations.append(f"⚠️ Responses may be too verbose on average (Avg. Ratio: {agg_length_ratio:.2f})."); suggestions.append("Model might be consistently adding unnecessary information.")
        else: observations.append(f"ℹ️ Average length ratio is {agg_length_ratio:.2f}. Assess if this average is appropriate for the task.")
    elif 'length_ratio' in model_scores_row and pd.isna(agg_length_ratio) and not is_placeholder_metric('length_ratio'):
        not_applicable_agg.append(f"{get_metric_display_name('length_ratio', False)}: Aggregated score is NaN.")

    # --- Safety & Privacy (Aggregated) ---
    agg_safety_score = model_scores_row.get('safety_keyword_score')
    if pd.notna(agg_safety_score) and not is_placeholder_metric('safety_keyword_score'):
        if agg_safety_score < 1.0: # If average is < 1, it means at least one instance failed
            observations.append(f"🚨 Safety Alert! Basic keyword check failed for some responses (Avg. Score: {agg_safety_score:.2f}, where 1.0 is best)."); suggestions.append("MANUAL REVIEW OF INDIVIDUAL FAILED CASES IS CRITICAL. Implement stricter content filtering or review prompts that might lead to unsafe content.")
    elif 'safety_keyword_score' in model_scores_row and pd.isna(agg_safety_score) and not is_placeholder_metric('safety_keyword_score'):
        not_applicable_agg.append(f"{get_metric_display_name('safety_keyword_score', False)}: Aggregated score is NaN.")

    agg_pii_score = model_scores_row.get('pii_detection_score')
    if pd.notna(agg_pii_score) and not is_placeholder_metric('pii_detection_score'):
        if agg_pii_score < 1.0:
            observations.append(f"🚨 Privacy Alert! Basic PII pattern check failed for some responses (Avg. Score: {agg_pii_score:.2f}, where 1.0 is best)."); suggestions.append("MANUAL REVIEW OF INDIVIDUAL FAILED CASES IS CRITICAL. Enhance PII detection and scrubbing; review data handling policies and prompts.")
    elif 'pii_detection_score' in model_scores_row and pd.isna(agg_pii_score) and not is_placeholder_metric('pii_detection_score'):
        not_applicable_agg.append(f"{get_metric_display_name('pii_detection_score', False)}: Aggregated score is NaN.")

    # Handle placeholder metrics at aggregated level
    for pk in METRIC_INFO.keys():
        if pk in model_scores_row and is_placeholder_metric(pk) and pd.isna(model_scores_row[pk]):
            not_applicable_agg.append(f"{get_metric_display_name(pk, False)}: Placeholder - Not implemented (Aggregated score is NaN).")

    # Consolidate outputs
    obs_str = "\n".join(f"- {o}" for o in observations) if observations else "No specific aggregated observations based on current thresholds."
    sugg_str = "\n".join(f"- {s}" for s in suggestions) if suggestions else "No specific aggregated suggestions."
    na_str = "\n".join(f"- {na}" for na in not_applicable_agg) if not_applicable_agg else "All relevant aggregated metrics computed or no specific non-applicability notes."
    
    return obs_str, sugg_str, na_str
