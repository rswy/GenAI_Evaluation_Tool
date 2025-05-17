# src/interpretation_engine.py
"""
Handles the logic for generating textual interpretations of metric scores
for both individual test cases and aggregated results.
"""
import numpy as np
import pandas as pd

# Import config and helpers
from .app_config import METRIC_INFO, SEMANTIC_SIMILARITY_SCORE, CLASSIFICATION # Assuming CLASSIFICATION constant is defined in app_config
from .ui_helpers import get_metric_display_name, is_placeholder_metric

def generate_single_case_interpretation(case_row, task_type):
    """
    Generates observations, suggestions, and notes on non-applicable metrics
    for a single test case row.
    Args:
        case_row (pd.Series): A row from the individual scores DataFrame.
        task_type (str): The task type for this case.
    Returns:
        tuple: (observations_str, suggestions_str, not_applicable_str)
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
            return input_key in case_data and pd.notna(case_data[input_key]) and str(case_data[input_key]).strip()
        return True

    # --- Fluency & Lexical Similarity ---
    fluency_metric_keys = ['bleu', 'rouge_1', 'rouge_2', 'rouge_l', 'meteor']
    valid_fluency_scores = []
    for key in fluency_metric_keys:
        if key in case_row and pd.notna(case_row[key]) and not is_placeholder_metric(key):
            valid_fluency_scores.append(case_row[key])
        elif key in case_row and pd.isna(case_row[key]) and not is_placeholder_metric(key):
            not_applicable_metrics.append(f"{get_metric_display_name(key, False)}: Score is NaN.")
    if valid_fluency_scores:
        avg_fluency = np.mean(valid_fluency_scores)
        if avg_fluency >= 0.6: observations.append(f"✅ Fluency & Lexical Sim.: Strong (Avg. Score: {avg_fluency:.2f})."); overall_assessment_flags["fluency"] = "good"
        elif avg_fluency >= 0.3: observations.append(f"⚠️ Fluency & Lexical Sim.: Moderate (Avg. Score: {avg_fluency:.2f})."); overall_assessment_flags["fluency"] = "moderate"
        else: observations.append(f"❌ Fluency & Lexical Sim.: Low (Avg. Score: {avg_fluency:.2f})."); overall_assessment_flags["fluency"] = "poor"; suggestions.append("Lexical similarity is low. Review for grammar, coherence. Compare with Semantic Similarity.")

    # --- Semantic Understanding ---
    metric_key_sem_sim = SEMANTIC_SIMILARITY_SCORE
    score_sem_sim = case_row.get(metric_key_sem_sim)
    if metric_key_sem_sim in case_row: # Check if column exists
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
    score_fp = case_row.get(metric_key_fp)
    if metric_key_fp in case_row:
        if not was_input_provided(metric_key_fp, case_row): not_applicable_metrics.append(f"{get_metric_display_name(metric_key_fp, False)}: Not applicable (missing `ref_facts`).")
        elif pd.notna(score_fp):
            if score_fp >= 0.7: observations.append(f"✅ Fact Presence: Good ({score_fp:.2f}). Most specified facts included."); overall_assessment_flags["factuality"] = "good"
            elif score_fp >= 0.4: observations.append(f"⚠️ Fact Presence: Moderate ({score_fp:.2f}). Some facts missing/altered."); overall_assessment_flags["factuality"] = "moderate"
            else: observations.append(f"❌ Fact Presence: Low ({score_fp:.2f}). Significant facts missing."); overall_assessment_flags["factuality"] = "poor"; suggestions.append("Verify missing/inaccurate critical facts from `ref_facts`.")
        elif pd.isna(score_fp): not_applicable_metrics.append(f"{get_metric_display_name(metric_key_fp, False)}: Score is NaN (Input `ref_facts` provided; review calculation).")

    # --- Completeness & Coverage (Key Point Coverage) ---
    metric_key_cc = 'completeness_score'
    score_cc = case_row.get(metric_key_cc)
    if metric_key_cc in case_row:
        if not was_input_provided(metric_key_cc, case_row): not_applicable_metrics.append(f"{get_metric_display_name(metric_key_cc, False)}: Not applicable (missing `ref_key_points`).")
        elif pd.notna(score_cc):
            if score_cc >= 0.7: observations.append(f"✅ Key Point Coverage: Good ({score_cc:.2f}). Most key points covered."); overall_assessment_flags["completeness"] = "good"
            elif score_cc >= 0.4: observations.append(f"⚠️ Key Point Coverage: Moderate ({score_cc:.2f}). Some key points unaddressed."); overall_assessment_flags["completeness"] = "moderate"
            else: observations.append(f"❌ Key Point Coverage: Low ({score_cc:.2f}). Significant key points missing."); overall_assessment_flags["completeness"] = "poor"; suggestions.append("Verify if essential topics from `ref_key_points` were addressed.")
        elif pd.isna(score_cc): not_applicable_metrics.append(f"{get_metric_display_name(metric_key_cc, False)}: Score is NaN (Input `ref_key_points` provided; review calculation).")

    # --- Classification Accuracy ---
    if task_type == CLASSIFICATION: # Use constant from app_config
        accuracy = case_row.get('accuracy')
        if 'accuracy' in case_row and pd.notna(accuracy):
            if accuracy < 1.0: observations.append(f"❌ Classification: Incorrect (Accuracy: {accuracy:.2f})."); overall_assessment_flags["classification"] = "incorrect"; suggestions.append("Analyze misclassification. Review input/ground truth.")
            else: observations.append(f"✅ Classification: Correct (Accuracy: {accuracy:.2f})."); overall_assessment_flags["classification"] = "correct"
        elif 'accuracy' in case_row and pd.isna(accuracy): not_applicable_metrics.append(f"{get_metric_display_name('accuracy', False)}: Score is NaN.")

    # --- Conciseness ---
    length_ratio = case_row.get('length_ratio')
    if 'length_ratio' in case_row and pd.notna(length_ratio):
        if length_ratio < 0.5: observations.append(f"⚠️ Conciseness: Response significantly shorter (ratio: {length_ratio:.2f})."); overall_assessment_flags["conciseness"] = "too_short"; suggestions.append("Check if response is overly brief/truncated.")
        elif length_ratio < 0.8: observations.append(f"ℹ️ Conciseness: Response noticeably shorter (ratio: {length_ratio:.2f})."); overall_assessment_flags["conciseness"] = "short"
        elif length_ratio <= 1.25: observations.append(f"✅ Conciseness: Response length comparable to reference (ratio: {length_ratio:.2f})."); overall_assessment_flags["conciseness"] = "good"
        elif length_ratio <= 1.75: observations.append(f"ℹ️ Conciseness: Response noticeably longer (ratio: {length_ratio:.2f})."); overall_assessment_flags["conciseness"] = "long"
        else: observations.append(f"⚠️ Conciseness: Response significantly longer (ratio: {length_ratio:.2f})."); overall_assessment_flags["conciseness"] = "too_long"; suggestions.append("Check if response is too verbose/irrelevant.")
    elif 'length_ratio' in case_row and pd.isna(length_ratio): not_applicable_metrics.append(f"{get_metric_display_name('length_ratio', False)}: Score is NaN.")

    # --- Safety & Privacy (Only show if issue detected) ---
    safety_score = case_row.get('safety_keyword_score')
    if 'safety_keyword_score' in case_row and pd.notna(safety_score):
        if safety_score < 1.0:
            observations.append("🚨 Safety: Potential safety keyword detected (Basic Check).")
            overall_assessment_flags["safety"] = "issue"
            suggestions.append("MANUAL REVIEW REQUIRED for safety. Identify problematic content; refine safety filters/prompts.")
    elif 'safety_keyword_score' in case_row and pd.isna(safety_score):
        not_applicable_metrics.append(f"{get_metric_display_name('safety_keyword_score', False)}: Score is NaN.")

    pii_score = case_row.get('pii_detection_score')
    if 'pii_detection_score' in case_row and pd.notna(pii_score):
        if pii_score < 1.0:
            observations.append("🚨 Privacy: Potential PII pattern detected (Basic Regex Check).")
            overall_assessment_flags["privacy"] = "issue"
            suggestions.append("MANUAL REVIEW REQUIRED for PII. Ensure sensitive data is not exposed; enhance PII detection/scrubbing.")
    elif 'pii_detection_score' in case_row and pd.isna(pii_score):
        not_applicable_metrics.append(f"{get_metric_display_name('pii_detection_score', False)}: Score is NaN.")

    # Handle placeholder metrics
    placeholder_keys_to_check = ["professional_tone_score", "refusal_quality_score", "nli_entailment_score", "llm_judge_factuality"]
    for pk in placeholder_keys_to_check:
        if pk in case_row:
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


def generate_aggregated_interpretations(model_scores_interp, task_type_interp):
    """
    Generates interpretations for a row of aggregated scores for a model/task.
    Args:
        model_scores_interp (pd.Series): Aggregated scores for one model and task.
        task_type_interp (str): The task type.
    Returns:
        tuple: (interpretations_list, suggestions_list)
    """
    interpretations = []
    suggestions_interp = []

    # Fluency
    fluency_keys = ['bleu', 'rouge_l', 'meteor']
    fluency_scores_interp = {m: model_scores_interp.get(m) for m in fluency_keys 
                             if m in model_scores_interp and pd.notna(model_scores_interp.get(m)) and not is_placeholder_metric(m)}
    if fluency_scores_interp:
        valid_fluency_scores_interp = [s for s in fluency_scores_interp.values() if pd.notna(s)]
        if valid_fluency_scores_interp:
            avg_fluency_interp = np.mean(valid_fluency_scores_interp)
            if avg_fluency_interp > 0.5: interpretations.append(f"✅ Generally good fluency & lexical similarity (Avg. relevant score: {avg_fluency_interp:.2f}).")
            elif avg_fluency_interp > 0.2: interpretations.append(f"⚠️ Moderate fluency/lexical similarity (Avg. score: {avg_fluency_interp:.2f}).")
            else: interpretations.append(f"❌ Low fluency/lexical similarity (Avg. score: {avg_fluency_interp:.2f})."); suggestions_interp.append("Review responses for clarity, grammar. Compare with semantic similarity.")

    # Semantic Similarity
    semantic_sim_interp = model_scores_interp.get(SEMANTIC_SIMILARITY_SCORE)
    if pd.notna(semantic_sim_interp) and not is_placeholder_metric(SEMANTIC_SIMILARITY_SCORE):
        if semantic_sim_interp > 0.7: interpretations.append(f"✅ Good semantic similarity to references ({semantic_sim_interp:.2f}). Meaning is well-aligned.")
        elif semantic_sim_interp > 0.4: interpretations.append(f"ℹ️ Moderate semantic similarity ({semantic_sim_interp:.2f}). Meaning somewhat aligned.")
        else: interpretations.append(f"⚠️ Low semantic similarity ({semantic_sim_interp:.2f}). Meaning may differ significantly."); suggestions_interp.append("Low semantic similarity can indicate misunderstanding of query or factual divergence.")

    # Fact Presence
    fact_presence_interp = model_scores_interp.get('fact_presence_score')
    if pd.notna(fact_presence_interp) and not is_placeholder_metric('fact_presence_score'):
        if fact_presence_interp > 0.7: interpretations.append(f"✅ Good inclusion of specified facts ({fact_presence_interp:.2f}).")
        elif fact_presence_interp > 0.4: interpretations.append(f"⚠️ Moderate inclusion of facts ({fact_presence_interp:.2f}).")
        else: interpretations.append(f"❌ Low inclusion of specified facts ({fact_presence_interp:.2f})."); suggestions_interp.append("Ensure `ref_facts` are accurate. Model might need better prompting for fact extraction.")
    elif pd.isna(fact_presence_interp) and 'fact_presence_score' in model_scores_interp and not is_placeholder_metric('fact_presence_score'):
        interpretations.append(f"ℹ️ Fact Presence: Not applicable or not computed (score is NaN). Likely due to missing `ref_facts` in input data.")

    # Key Point Coverage
    completeness_interp = model_scores_interp.get('completeness_score')
    if pd.notna(completeness_interp) and not is_placeholder_metric('completeness_score'):
        if completeness_interp > 0.7: interpretations.append(f"✅ Good coverage of key points ({completeness_interp:.2f}).")
        elif completeness_interp > 0.4: interpretations.append(f"⚠️ Moderate coverage of key points ({completeness_interp:.2f}).")
        else: interpretations.append(f"❌ Low coverage of key points ({completeness_interp:.2f})."); suggestions_interp.append("Ensure `ref_key_points` are well-defined. Model might need prompting for comprehensiveness.")
    elif pd.isna(completeness_interp) and 'completeness_score' in model_scores_interp and not is_placeholder_metric('completeness_score'):
         interpretations.append(f"ℹ️ Key Point Coverage: Not applicable or not computed (score is NaN). Likely due to missing `ref_key_points` in input data.")

    # Classification
    if task_type_interp == CLASSIFICATION:
        f1_interp = model_scores_interp.get('f1_score'); acc_interp = model_scores_interp.get('accuracy')
        if pd.notna(f1_interp) and not is_placeholder_metric('f1_score'):
            if f1_interp > 0.75: interpretations.append(f"✅ Good classification performance (F1: {f1_interp:.2f}).")
            elif f1_interp > 0.5: interpretations.append(f"⚠️ Moderate classification performance (F1: {f1_interp:.2f}).")
            else: interpretations.append(f"❌ Low classification performance (F1: {f1_interp:.2f})."); suggestions_interp.append("Review misclassified examples. Consider more/better training data.")
        elif pd.notna(acc_interp) and not is_placeholder_metric('accuracy'):
             if acc_interp > 0.75: interpretations.append(f"✅ Good classification accuracy ({acc_interp:.2f}).")
             else: interpretations.append(f"⚠️ Classification accuracy is {acc_interp:.2f}. Check F1/Precision/Recall."); suggestions_interp.append("Review misclassified examples if accuracy is low.")

    # Conciseness
    length_ratio_interp = model_scores_interp.get('length_ratio')
    if pd.notna(length_ratio_interp) and not is_placeholder_metric('length_ratio'):
        if 0.75 <= length_ratio_interp <= 1.25: interpretations.append(f"✅ Good response length relative to reference (Avg. Ratio: {length_ratio_interp:.2f}).")
        elif length_ratio_interp < 0.5: interpretations.append(f"⚠️ Responses may be too short on average (Avg. Ratio: {length_ratio_interp:.2f})."); suggestions_interp.append("Check if model is truncating answers or being overly brief.")
        elif length_ratio_interp > 1.75: interpretations.append(f"⚠️ Responses may be too verbose on average (Avg. Ratio: {length_ratio_interp:.2f})."); suggestions_interp.append("Model might be adding unnecessary information.")
        else: interpretations.append(f"ℹ️ Average length ratio is {length_ratio_interp:.2f}. Assess if appropriate for the task.")

    # Safety & Privacy
    safety_score_interp = model_scores_interp.get('safety_keyword_score')
    if pd.notna(safety_score_interp) and not is_placeholder_metric('safety_keyword_score'):
        if safety_score_interp < 1.0: interpretations.append(f"🚨 Safety alert! Basic keyword check failed for some responses (Avg Score: {safety_score_interp:.2f}). MANUAL REVIEW OF INDIVIDUAL CASES IS CRITICAL."); suggestions_interp.append("Implement stricter content filtering.")

    pii_score_interp = model_scores_interp.get('pii_detection_score')
    if pd.notna(pii_score_interp) and not is_placeholder_metric('pii_detection_score'):
        if pii_score_interp < 1.0: interpretations.append(f"🚨 Privacy alert! Basic PII pattern check failed for some responses (Avg Score: {pii_score_interp:.2f}). MANUAL REVIEW IS CRITICAL."); suggestions_interp.append("Enhance PII detection/scrubbing.")

    return interpretations, suggestions_interp
