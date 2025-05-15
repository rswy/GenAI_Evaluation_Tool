# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys
import os
import datetime
import tempfile
import json
import numpy as np
from collections import defaultdict
import warnings
import matplotlib # Keep for colormap
import matplotlib.cm as cm # Keep for colormap
import copy # Keep for deepcopy if used

# --- Add project root to sys.path ---
project_root = Path(__file__).resolve().parent # Use resolve()
data_dir = project_root / "data"
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# --- Import framework functions ---
try:
    from data_loader import load_data
    from evaluator import evaluate_model_responses # Returns two DataFrames now
    from file_converter import convert_excel_to_data, convert_csv_to_data
    from mock_data_generator import generate_mock_data_flat, save_mock_data
    from tasks.task_registry import get_metrics_for_task, get_supported_tasks, RAG_FAQ, SUMMARIZATION, CLASSIFICATION, CHATBOT
except ImportError as e:
    st.error(f"Framework Import Error: {e}. Please ensure all necessary files are in the 'src' directory and Python environment is set up correctly.")
    st.error(f"Current sys.path: {sys.path}")
    st.error(f"Project root evaluated as: {project_root}")
    st.stop()

# --- Metric Information (for display purposes) ---
CAT_TRUST = "Trust & Factuality"; CAT_COMPLETENESS = "Completeness"; CAT_FLUENCY = "Fluency & Similarity"
CAT_CLASSIFICATION = "Classification Accuracy"; CAT_CONCISENESS = "Conciseness"; CAT_SAFETY = "Safety"
CAT_PII_SAFETY = "Privacy/Sensitive Data"; CAT_TONE = "Tone & Professionalism"; CAT_REFUSAL = "Refusal Appropriateness"

DIMENSION_DESCRIPTIONS = {
    CAT_TRUST: "Metrics assessing the reliability and factual correctness of the LLM's output, aiming to minimize hallucinations and ensure grounding in provided contexts for RAG tasks.",
    CAT_COMPLETENESS: "Metrics evaluating if the LLM response addresses all necessary aspects or key points required by the input query or task instructions.",
    CAT_FLUENCY: "Metrics judging the linguistic quality of the LLM's output, including grammatical correctness, coherence, and similarity to human-like language.",
    CAT_CLASSIFICATION: "Metrics specifically for classification tasks, measuring the accuracy of the LLM in assigning correct labels or categories.",
    CAT_CONCISENESS: "Metrics gauging the brevity and focus of the LLM's response, preferring shorter, to-the-point answers where appropriate.",
    CAT_SAFETY: "Metrics performing basic checks for harmful, biased, or inappropriate content in the LLM's output.",
    CAT_PII_SAFETY: "Metrics focused on detecting Personal Identifiable Information (PII) or other sensitive data within the LLM's responses.",
    CAT_TONE: "Metrics (often placeholders) for assessing the professionalism, politeness, or other specific tonal qualities of the LLM's output.",
    CAT_REFUSAL: "Metrics (often placeholders) for evaluating the appropriateness of the LLM's refusals to answer certain queries, especially those that are out-of-scope, sensitive, or harmful."
}
METRIC_INFO = {
    "fact_presence_score": {"name": "Fact Presence", "category": CAT_TRUST, "higher_is_better": True, "explanation": "Checks if predefined factual statements (from `ref_facts` column) are mentioned in the model's answer. Higher score indicates more listed facts were found.", "tasks": [RAG_FAQ]},
    "completeness_score": {"name": "Checklist Completeness", "category": CAT_COMPLETENESS, "higher_is_better": True, "explanation": "Assesses if predefined key topics or items (from `ref_key_points` column) are mentioned in the model's answer. Higher score means more points were covered.", "tasks": [RAG_FAQ, SUMMARIZATION]},
    "bleu": {"name": "BLEU", "category": CAT_FLUENCY, "higher_is_better": True, "explanation": "Measures n-gram precision overlap between the model's answer and the ground truth, indicating sequence similarity. Higher score means more similar sequences.", "tasks": [RAG_FAQ, SUMMARIZATION, CHATBOT]},
    "rouge_1": {"name": "ROUGE-1", "category": CAT_FLUENCY, "higher_is_better": True, "explanation": "Measures unigram (single word) recall overlap. Higher score indicates more matching words with the ground truth.", "tasks": [RAG_FAQ, SUMMARIZATION, CHATBOT]},
    "rouge_2": {"name": "ROUGE-2", "category": CAT_FLUENCY, "higher_is_better": True, "explanation": "Measures bigram (two-word phrase) recall overlap. Higher score indicates more matching short phrases with the ground truth.", "tasks": [RAG_FAQ, SUMMARIZATION, CHATBOT]},
    "rouge_l": {"name": "ROUGE-L", "category": CAT_FLUENCY, "higher_is_better": True, "explanation": "Measures the longest common subsequence overlap, reflecting sentence-level structure similarity with the ground truth. Higher score means greater similarity.", "tasks": [RAG_FAQ, SUMMARIZATION, CHATBOT]},
    "meteor": {"name": "METEOR", "category": CAT_FLUENCY, "higher_is_better": True, "explanation": "Harmonic mean of unigram precision and recall, with stemming and synonym matching. Higher score indicates a better overall match with the ground truth.", "tasks": [RAG_FAQ, SUMMARIZATION, CHATBOT]},
    "accuracy": {"name": "Accuracy", "category": CAT_CLASSIFICATION, "higher_is_better": True, "explanation": "For classification tasks, the overall percentage of correct predictions. For individual cases, 1.0 if correct, 0.0 if incorrect.", "tasks": [CLASSIFICATION]},
    "precision": {"name": "Precision", "category": CAT_CLASSIFICATION, "higher_is_better": True, "explanation": "Of positive predictions, the fraction that were correct. Higher score means fewer false positives. (Note: Aggregated score is standard; individual is 1.0/0.0 for the pair based on a specific class perspective).", "tasks": [CLASSIFICATION]},
    "recall": {"name": "Recall", "category": CAT_CLASSIFICATION, "higher_is_better": True, "explanation": "Of actual positives, the fraction correctly predicted. Higher score means fewer false negatives. (Note: Aggregated score is standard; individual is 1.0/0.0 for the pair based on a specific class perspective).", "tasks": [CLASSIFICATION]},
    "f1_score": {"name": "F1-Score", "category": CAT_CLASSIFICATION, "higher_is_better": True, "explanation": "Harmonic mean of Precision and Recall, providing a balance between them. (Note: Aggregated score is standard; individual is 1.0/0.0 for the pair based on a specific class perspective).", "tasks": [CLASSIFICATION]},
    "length_ratio": {"name": "Length Ratio", "category": CAT_CONCISENESS, "higher_is_better": False, "explanation": "Ratio of the model answer's word count to the ground truth's word count. Closer to 1.0 is often desired; significantly higher or lower may indicate verbosity or brevity issues.", "tasks": [RAG_FAQ, SUMMARIZATION, CHATBOT]},
    "safety_keyword_score": {"name": "Safety Keyword Score", "category": CAT_SAFETY, "higher_is_better": True, "explanation": "Basic check: 1.0 if response does NOT contain predefined unsafe keywords, 0.0 otherwise. Higher means fewer problematic keywords detected.", "tasks": [RAG_FAQ, SUMMARIZATION, CHATBOT]},
    "pii_detection_score": {"name": "PII Detection Score", "category": CAT_PII_SAFETY, "higher_is_better": True, "explanation": "Basic regex check: 1.0 if response does NOT contain common PII patterns, 0.0 otherwise. Higher means fewer PII patterns found. WARNING: Not a comprehensive PII scan.", "tasks": [RAG_FAQ, CHATBOT]},
    "professional_tone_score": {"name": "Professional Tone (Placeholder)", "category": CAT_TONE, "higher_is_better": True, "explanation": "Placeholder for professional tone evaluation. Requires a dedicated classifier or LLM-as-judge implementation.", "tasks": [RAG_FAQ, CHATBOT]},
    "refusal_quality_score": {"name": "Refusal Quality (Placeholder)", "category": CAT_REFUSAL, "higher_is_better": True, "explanation": "Placeholder for evaluating the appropriateness of model refusals. Requires specific test cases and logic.", "tasks": [RAG_FAQ, CHATBOT]},
    "nli_entailment_score": {"name": "NLI Entailment (Placeholder)", "category": CAT_TRUST, "higher_is_better": True, "explanation": "Placeholder for Natural Language Inference based fact-checking or groundedness. Requires an NLI model.", "tasks": [RAG_FAQ]},
    "llm_judge_factuality": {"name": "LLM Judge Factuality (Placeholder)", "category": CAT_TRUST, "higher_is_better": True, "explanation": "Placeholder for using another LLM to judge factuality. Requires LLM API access and prompt engineering.", "tasks": [RAG_FAQ]},
}
METRICS_BY_CATEGORY = defaultdict(list)
CATEGORY_ORDER = [CAT_TRUST, CAT_COMPLETENESS, CAT_FLUENCY, CAT_CLASSIFICATION, CAT_CONCISENESS, CAT_SAFETY, CAT_PII_SAFETY, CAT_TONE, CAT_REFUSAL]
for key, info in METRIC_INFO.items(): METRICS_BY_CATEGORY[info['category']].append(key)

def get_metric_display_name(metric_key, include_placeholder_tag=True):
    """Gets the display name for a metric, optionally including a placeholder tag."""
    info = METRIC_INFO.get(metric_key, {})
    name = info.get('name', metric_key.replace('_', ' ').title())
    if include_placeholder_tag and is_placeholder_metric(metric_key): 
        if "(Placeholder)" not in name: # Avoid double-tagging
            name += " (Placeholder)"
    return name

def get_metric_indicator(metric_key):
    info = METRIC_INFO.get(metric_key); return "⬆️" if info and info["higher_is_better"] else ("⬇️" if info else "")

def is_placeholder_metric(metric_key):
    """Checks if a metric is a placeholder based on its name in METRIC_INFO."""
    info = METRIC_INFO.get(metric_key, {})
    return "(Placeholder)" in info.get("name", "")


def apply_color_gradient(styler, metric_info_dict_local):
    cmap_good = matplotlib.colormaps.get_cmap('RdYlGn')
    cmap_bad = matplotlib.colormaps.get_cmap('RdYlGn_r')
    
    for col_name_display in styler.columns: 
        if not isinstance(col_name_display, str):
            continue

        original_metric_key = None
        for mk_orig, m_info_orig in metric_info_dict_local.items():
            possible_names = [
                f"{get_metric_display_name(mk_orig, True)} {get_metric_indicator(mk_orig)}".strip(),
                f"{get_metric_display_name(mk_orig, False)} {get_metric_indicator(mk_orig)}".strip(),
                get_metric_display_name(mk_orig, True).strip(),
                get_metric_display_name(mk_orig, False).strip()
            ]
            if col_name_display.strip() in possible_names:
                original_metric_key = mk_orig
                break
        
        info = metric_info_dict_local.get(original_metric_key)

        if info and pd.api.types.is_numeric_dtype(styler.data[col_name_display]):
            cmap_to_use = cmap_good if info['higher_is_better'] else cmap_bad
            try:
                data_col = styler.data[col_name_display].dropna()
                vmin = data_col.min() if not data_col.empty else 0.0
                vmax = data_col.max() if not data_col.empty else 1.0
                
                if vmin == vmax:
                    mid_point = 0.5 
                    color_val = 0.0 
                    if info['higher_is_better']:
                        if vmin > mid_point: color_val = 1.0 
                        elif np.isclose(vmin, mid_point): color_val = 0.5 
                    else: 
                        if vmin < mid_point: color_val = 1.0 
                        elif np.isclose(vmin, mid_point): color_val = 0.5 
                    styler.background_gradient(cmap=matplotlib.colors.ListedColormap([cmap_to_use(color_val)]), subset=[col_name_display])
                else:
                    gradient_vmin = min(0.0, vmin) if pd.notna(vmin) else 0.0
                    gradient_vmax = max(1.0, vmax) if pd.notna(vmax) else 1.0
                    if gradient_vmin == gradient_vmax: 
                         styler.background_gradient(cmap=matplotlib.colors.ListedColormap([cmap_to_use(0.5 if gradient_vmin == 0.5 else (1.0 if gradient_vmin > 0.5 else 0.0))]), subset=[col_name_display])
                    else:
                         styler.background_gradient(cmap=cmap_to_use, subset=[col_name_display], vmin=gradient_vmin, vmax=gradient_vmax)
            except Exception as e:
                warnings.warn(f"Style error for '{col_name_display}' (orig key: {original_metric_key}): {e}", RuntimeWarning)
    
    format_dict = {}
    for col_disp_name in styler.columns: 
        if isinstance(col_disp_name, str):
            original_mkey_for_format = None
            for mk_orig, m_info_orig in metric_info_dict_local.items():
                possible_names = [
                    f"{get_metric_display_name(mk_orig, True)} {get_metric_indicator(mk_orig)}".strip(),
                    f"{get_metric_display_name(mk_orig, False)} {get_metric_indicator(mk_orig)}".strip(),
                    get_metric_display_name(mk_orig, True).strip(),
                    get_metric_display_name(mk_orig, False).strip()
                ]
                if col_disp_name.strip() in possible_names:
                    original_mkey_for_format = mk_orig
                    break
            
            if original_mkey_for_format and pd.api.types.is_numeric_dtype(styler.data[col_disp_name]):
                 format_dict[col_disp_name] = '{:.4f}'
            # Check if it's one of the new interpretation columns, which should be displayed as strings
            elif col_disp_name in ['Observations', 'Potential Actions']:
                pass # No specific numeric formatting
            elif styler.data[col_disp_name].dtype == 'object' and any(isinstance(x, float) for x in styler.data[col_disp_name].dropna()):
                 format_dict[col_disp_name] = lambda x: f"{x:.4f}" if isinstance(x, float) else x


    styler.format(formatter=format_dict, na_rep='NaN')
    return styler

def unflatten_df_to_test_cases(df):
    test_cases_list = []
    if df is None or df.empty: return []
    direct_keys = ['id', 'task_type', 'model', 'question', 'contexts', 'ground_truth', 'answer', 'ref_facts', 'ref_key_points', 'test_description']
    for _, row_series in df.iterrows():
        row = row_series.to_dict()
        case = {}
        if pd.isna(row.get('task_type')) or pd.isna(row.get('model')) or \
           pd.isna(row.get('question')) or pd.isna(row.get('ground_truth')) or \
           pd.isna(row.get('answer')):
            st.warning(f"Skipping row due to missing required field(s) (task_type, model, question, ground_truth, answer): {row.get('id', 'Unknown ID')}")
            continue
        for key in direct_keys:
            case[key] = str(row[key]) if key in row and pd.notna(row[key]) else None
        for col_name, value in row.items():
            if col_name not in case: 
                case[col_name] = str(value) if pd.notna(value) else None
        test_cases_list.append(case)
    return test_cases_list

def generate_single_case_interpretation(case_row, task_type):
    """Generates observations and suggestions for a single test case row."""
    observations = []
    suggestions = []
    
    # Fluency & Similarity
    fluency_scores_present = {m: case_row.get(m) for m in ['bleu', 'rouge_l', 'meteor'] if m in case_row and pd.notna(case_row.get(m)) and not is_placeholder_metric(m)}
    if fluency_scores_present:
        # Calculate mean only if there are valid (non-NaN) scores
        valid_fluency_scores = [s for s in fluency_scores_present.values() if pd.notna(s)]
        if valid_fluency_scores:
            avg_fluency = np.mean(valid_fluency_scores)
            if avg_fluency < 0.2: 
                observations.append(f"Low fluency/similarity to reference (avg: {avg_fluency:.2f}).")
                suggestions.append("Review for linguistic issues or significant deviation from reference phrasing.")
            elif avg_fluency < 0.5:
                observations.append(f"Moderate fluency/similarity (avg: {avg_fluency:.2f}).")
        elif any(pd.notna(s) for s in fluency_scores_present.values()): # Some scores present but all were NaN after filtering
             observations.append("Fluency scores present but NaN, check metric calculation for this case.")


    # Trust & Factuality
    fact_presence = case_row.get('fact_presence_score')
    if pd.notna(fact_presence) and not is_placeholder_metric('fact_presence_score'):
        if fact_presence < 0.4: 
            observations.append(f"Low inclusion of specified facts ({fact_presence:.2f}).")
            suggestions.append("Check if critical facts from `ref_facts` are missing in the answer.")
        elif fact_presence < 0.7:
            observations.append(f"Moderate inclusion of facts ({fact_presence:.2f}).")

    # Completeness
    completeness = case_row.get('completeness_score')
    if pd.notna(completeness) and not is_placeholder_metric('completeness_score'):
        if completeness < 0.4: 
            observations.append(f"Low coverage of key points ({completeness:.2f}).")
            suggestions.append("Verify if all essential topics from `ref_key_points` were addressed.")
        elif completeness < 0.7:
            observations.append(f"Moderate coverage of key points ({completeness:.2f}).")

    # Classification (Accuracy is 1 or 0 for individual)
    accuracy = case_row.get('accuracy')
    if task_type == CLASSIFICATION and pd.notna(accuracy) and not is_placeholder_metric('accuracy'):
        if accuracy < 1.0: # essentially if accuracy is 0 for the instance
            observations.append("Incorrect classification.")
            suggestions.append("Analyze why this specific case was misclassified.")
    
    # Conciseness
    length_ratio = case_row.get('length_ratio')
    if pd.notna(length_ratio) and not is_placeholder_metric('length_ratio'):
        if length_ratio < 0.5: 
            observations.append(f"Response significantly shorter than reference (Ratio: {length_ratio:.2f}).")
            suggestions.append("Check if response is too brief or truncated.")
        elif length_ratio > 1.75: 
            observations.append(f"Response significantly longer than reference (Ratio: {length_ratio:.2f}).")
            suggestions.append("Check if response is too verbose or includes irrelevant details.")

    # Safety
    safety_score = case_row.get('safety_keyword_score')
    if pd.notna(safety_score) and not is_placeholder_metric('safety_keyword_score') and safety_score < 1.0:
        observations.append("Potential safety keyword detected.")
        suggestions.append("MANUAL REVIEW REQUIRED for safety.")
    
    pii_score = case_row.get('pii_detection_score')
    if pd.notna(pii_score) and not is_placeholder_metric('pii_detection_score') and pii_score < 1.0:
        observations.append("Potential PII pattern detected.")
        suggestions.append("MANUAL REVIEW REQUIRED for PII.")

    if not observations: observations.append("No immediate concerns based on these specific metrics for this case.")
    if not suggestions: suggestions.append("No specific automated suggestions for this case. Review manually if scores are unexpectedly low.")
    return "\n".join(f"- {o}" for o in observations), "\n".join(f"- {s}" for s in suggestions)


# --- Streamlit App Configuration ---
st.set_page_config(layout="wide", page_title="LLM Evaluation Framework")
st.title("📊 LLM Evaluation Framework")
st.markdown("Evaluate LLM performance using pre-generated responses. This tool now supports both **aggregated summaries** and **individual test case scores**.")

# --- State Management ---
default_state_keys = {
    'test_cases_list_loaded': None,
    'edited_test_cases_df': pd.DataFrame(),
    'aggregated_results_df': None,
    'individual_scores_df': None, 
    'data_source_info': None,
    'last_uploaded_file_name': None,
    'metrics_for_agg_display': [] 
}
for key, default_value in default_state_keys.items():
    if key not in st.session_state:
        st.session_state[key] = copy.deepcopy(default_value)

# --- Sidebar ---
st.sidebar.header("⚙️ Input Options")
def clear_app_state():
    st.session_state.test_cases_list_loaded = None
    st.session_state.edited_test_cases_df = pd.DataFrame()
    st.session_state.aggregated_results_df = None
    st.session_state.individual_scores_df = None
    st.session_state.data_source_info = None
    st.session_state.metrics_for_agg_display = []

input_method = st.sidebar.radio(
    "Choose data source:",
    ("Upload File", "Generate Mock Data"),
    key="input_method_radio",
    on_change=clear_app_state
)

if input_method == "Upload File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload (.xlsx, .csv, .json - Flat Format)",
        type=["xlsx", "csv", "json"],
        key="file_uploader"
    )
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.last_uploaded_file_name:
            clear_app_state()
            st.session_state.last_uploaded_file_name = uploaded_file.name
            file_suffix = Path(uploaded_file.name).suffix.lower()
            st.session_state.data_source_info = f"Processing: {uploaded_file.name}"
            st.sidebar.info(st.session_state.data_source_info)
            tmp_file_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = Path(tmp_file.name)
                test_data_list_from_file = None
                with st.spinner(f"Loading and converting {file_suffix}..."):
                    if file_suffix == ".xlsx": test_data_list_from_file = convert_excel_to_data(tmp_file_path)
                    elif file_suffix == ".csv": test_data_list_from_file = convert_csv_to_data(tmp_file_path)
                    elif file_suffix == ".json": test_data_list_from_file = load_data(tmp_file_path)
                if test_data_list_from_file:
                    st.session_state.test_cases_list_loaded = test_data_list_from_file
                    df_for_edit = pd.DataFrame(test_data_list_from_file)
                    required_cols_editor = ['id', 'task_type', 'model', 'question', 'contexts', 'ground_truth', 'answer', 'ref_facts', 'ref_key_points', 'test_description']
                    for col in required_cols_editor:
                        if col not in df_for_edit.columns: df_for_edit[col] = None
                    st.session_state.edited_test_cases_df = df_for_edit.copy().fillna('')
                    st.session_state.data_source_info = f"Loaded {len(test_data_list_from_file)} rows from {uploaded_file.name} into editor."
                    st.sidebar.success(st.session_state.data_source_info)
                else:
                    if test_data_list_from_file == []: raise ValueError("File loaded but was empty or contained no valid data.")
                    else: raise ValueError("Failed to load or convert data. Check file format and content.")
            except Exception as e:
                st.session_state.data_source_info = f"Error processing {uploaded_file.name}: {e}"
                st.sidebar.error(st.session_state.data_source_info); clear_app_state()
            finally:
                if tmp_file_path and tmp_file_path.exists():
                    try: os.unlink(tmp_file_path)
                    except Exception as e_unlink: warnings.warn(f"Could not delete temp file {tmp_file_path}: {e_unlink}")
elif input_method == "Generate Mock Data":
    st.sidebar.warning("Mock data provides example flat-format rows.")
    if st.sidebar.button("Generate and Use Mock Data", key="generate_mock_button"):
        clear_app_state()
        try:
            with st.spinner("Generating mock evaluation data..."):
                mock_data_list = generate_mock_data_flat(num_samples_per_task=3)
            if mock_data_list:
                st.session_state.test_cases_list_loaded = mock_data_list
                df_for_edit = pd.DataFrame(mock_data_list)
                required_cols_editor = ['id', 'task_type', 'model', 'question', 'contexts', 'ground_truth', 'answer', 'ref_facts', 'ref_key_points', 'test_description']
                for col in required_cols_editor:
                    if col not in df_for_edit.columns: df_for_edit[col] = None
                st.session_state.edited_test_cases_df = df_for_edit.copy().fillna('')
                st.session_state.data_source_info = f"Using {len(mock_data_list)} generated mock rows, loaded into editor."
                st.sidebar.success(st.session_state.data_source_info)
            else: st.sidebar.error("Failed to generate mock data.")
        except Exception as e:
            st.sidebar.error(f"Error generating mock data: {e}")
            st.sidebar.text_area("Traceback", traceback.format_exc(), height=150)

# --- Main Content Area ---
if st.session_state.data_source_info:
    if "error" in st.session_state.data_source_info.lower() or "failed" in st.session_state.data_source_info.lower() :
        st.error(st.session_state.data_source_info)
    elif "loaded" in st.session_state.data_source_info.lower() or "using" in st.session_state.data_source_info.lower():
        st.success(st.session_state.data_source_info)
    else: st.info(st.session_state.data_source_info)

tab_eval, tab_data_editor, tab_format_guide, tab_metrics_tutorial = st.tabs(["📊 Evaluation & Results", "📝 View/Edit/Add Data", "📄 Data Format Guide", "📖 Metrics Tutorial"])

with tab_eval:
    st.header("Run Evaluation and View Results")
    run_button_disabled = not isinstance(st.session_state.edited_test_cases_df, pd.DataFrame) or st.session_state.edited_test_cases_df.empty
    if st.button("🚀 Run Evaluation on Data in Editor", disabled=run_button_disabled, key="run_eval_main_button", help="Evaluates the current data shown in the 'View/Edit/Add Data' tab."):
        if isinstance(st.session_state.edited_test_cases_df, pd.DataFrame) and not st.session_state.edited_test_cases_df.empty:
            st.session_state.aggregated_results_df = None; st.session_state.individual_scores_df = None
            st.session_state.metrics_for_agg_display = []
            with st.spinner("⏳ Evaluating... This may take a moment."):
                try:
                    df_to_process = st.session_state.edited_test_cases_df.replace('', np.nan)
                    test_cases_to_evaluate = unflatten_df_to_test_cases(df_to_process)
                    if not test_cases_to_evaluate: raise ValueError("Data in editor is empty or could not be processed.")
                    
                    individual_df_raw, aggregated_df = evaluate_model_responses(test_cases_to_evaluate)
                    
                    if individual_df_raw is not None and not individual_df_raw.empty:
                        interpretations_series = individual_df_raw.apply(
                            lambda row: generate_single_case_interpretation(row, row.get('task_type')), axis=1
                        )
                        individual_df_raw['Observations'] = interpretations_series.apply(lambda x: x[0])
                        individual_df_raw['Potential Actions'] = interpretations_series.apply(lambda x: x[1])
                    
                    st.session_state.individual_scores_df = individual_df_raw 
                    st.session_state.aggregated_results_df = aggregated_df

                    if aggregated_df is not None and not aggregated_df.empty:
                        metrics_present_in_agg = [col for col in aggregated_df.columns if col in METRIC_INFO] 
                        metrics_to_show_agg = []
                        for metric_col in metrics_present_in_agg: 
                            if pd.api.types.is_numeric_dtype(aggregated_df[metric_col]):
                                is_all_zero_or_nan = (aggregated_df[metric_col].isna() | (aggregated_df[metric_col].abs() < 1e-9)).all()
                                if not is_all_zero_or_nan: metrics_to_show_agg.append(metric_col)
                        st.session_state.metrics_for_agg_display = metrics_to_show_agg 
                        st.success("✅ Evaluation complete! View results below.")
                    elif individual_df_raw is not None and not individual_df_raw.empty:
                         st.warning("⚠️ Evaluation produced individual scores, but aggregated results are empty.")
                    else: st.warning("⚠️ Evaluation finished, but no results were produced.")
                except Exception as e:
                     st.error(f"Evaluation error: {e}"); st.error(f"Traceback: {traceback.format_exc()}")
        else: st.warning("No data in editor. Load or generate data first.")

    st.divider(); st.header("Evaluation Results")
    res_tab_ind, res_tab_agg = st.tabs(["📄 Individual Scores","📈 Aggregated Results"])

    with res_tab_agg:
        # st.subheader("Aggregated Scores per Task & Model")

        st.success("Why Aggregated? \n Aggregated view transforms a collection of individual data points into meaningful insights about the LLM's overall behavior, strengths, weaknesses, and potential biases. It provides the necessary context to make informed decisions about model development, deployment, and ongoing monitoring.")
        if st.session_state.aggregated_results_df is not None and not st.session_state.aggregated_results_df.empty:
            agg_df = st.session_state.aggregated_results_df
            
            metrics_to_display_non_placeholder = [
                m_key for m_key in st.session_state.metrics_for_agg_display if not is_placeholder_metric(m_key)
            ]

            if not metrics_to_display_non_placeholder:
                st.info("No non-placeholder metrics with valid scores to display in the aggregated summary. Displaying raw aggregated data if available (may include placeholders or all-zero/NaN metrics).")
                simple_formatter = {col: "{:.4f}" for col in agg_df.select_dtypes(include=np.number).columns}
                st.dataframe(agg_df.style.format(formatter=simple_formatter, na_rep='NaN'), use_container_width=True)
            else:
                st.markdown("#### 🏆 Best Model Summary (Highlights)")
                st.caption("Top models per task based on key, non-placeholder metrics (non-zero scores only). Expand tasks.")
                key_metrics_per_task = {
                    RAG_FAQ: [m for m in ["fact_presence_score", "completeness_score", "rouge_l"] if not is_placeholder_metric(m)],
                    SUMMARIZATION: [m for m in ["completeness_score", "rouge_l", "length_ratio"] if not is_placeholder_metric(m)],
                    CLASSIFICATION: [m for m in ["f1_score", "accuracy"] if not is_placeholder_metric(m)],
                    CHATBOT: [m for m in ["meteor", "rouge_l", "length_ratio"] if not is_placeholder_metric(m)]
                }
                available_tasks_for_best_model = sorted(agg_df['task_type'].unique()) if 'task_type' in agg_df else []
                if not available_tasks_for_best_model: st.info("Run evaluation to see best model summaries.")
                else:
                    for task_type_bm in available_tasks_for_best_model:
                        with st.expander(f"**Task: {task_type_bm}**", expanded=False):
                            task_df_bm = agg_df[agg_df['task_type'] == task_type_bm].copy()
                            best_performers_details = []
                            current_task_key_metrics = [
                                m for m in key_metrics_per_task.get(task_type_bm, []) 
                                if m in task_df_bm.columns and pd.api.types.is_numeric_dtype(task_df_bm[m]) and m in metrics_to_display_non_placeholder
                            ]
                            if not current_task_key_metrics:
                                st.markdown("_No key, non-placeholder metrics available for highlights in this task._")
                                continue
                            for metric_bm in current_task_key_metrics:
                                info_bm = METRIC_INFO.get(metric_bm, {})
                                if not info_bm : continue 
                                higher_better_bm = info_bm.get('higher_is_better', True)
                                valid_scores_df_bm = task_df_bm.dropna(subset=[metric_bm])
                                if valid_scores_df_bm.empty: continue
                                best_score_idx_bm = valid_scores_df_bm[metric_bm].idxmax() if higher_better_bm else valid_scores_df_bm[metric_bm].idxmin()
                                best_row_bm = valid_scores_df_bm.loc[best_score_idx_bm]
                                best_model_bm = best_row_bm['model']
                                best_score_val_bm = best_row_bm[metric_bm]
                                if pd.notna(best_score_val_bm) and not np.isclose(best_score_val_bm, 0.0, atol=1e-9):
                                    best_performers_details.append({
                                        "metric_name_display": f"{get_metric_display_name(metric_bm)} {get_metric_indicator(metric_bm)}", 
                                        "model": best_model_bm, "score": best_score_val_bm,
                                        "explanation": info_bm.get('explanation', 'N/A')})
                            if not best_performers_details: st.markdown("_No significant highlights determined for this task._")
                            else:
                                highlights_by_model = defaultdict(list)
                                for detail in best_performers_details: highlights_by_model[detail["model"]].append(detail)
                                for model_name_bm, model_highlights in sorted(highlights_by_model.items()):
                                    st.markdown(f"**`{model_name_bm}` was best for:**")
                                    for highlight in model_highlights:
                                        st.markdown(f"- {highlight['metric_name_display']}: **{highlight['score']:.4f}**")
                                        st.caption(f"  *Metric Meaning:* {highlight['explanation']}")
                                    st.markdown("---") 
                    st.markdown("---") 

                st.markdown("#### 📊 Overall Summary Table (Aggregated by Task & Dimension)")
                agg_df_display_overall = agg_df.copy()
                renamed_cols_overall = {}
                final_display_cols_overall = []
                static_cols_display = ['task_type', 'model', 'num_samples']
                for static_col in static_cols_display:
                    if static_col in agg_df_display_overall.columns:
                        new_name = static_col.replace('_', ' ').title()
                        renamed_cols_overall[static_col] = new_name
                        final_display_cols_overall.append(new_name)
                
                for original_metric_key in metrics_to_display_non_placeholder: 
                    if original_metric_key in agg_df_display_overall.columns: 
                        indicator = get_metric_indicator(original_metric_key)
                        col_title = get_metric_display_name(original_metric_key, include_placeholder_tag=True) 
                        new_name = f"{col_title} {indicator}".strip()
                        renamed_cols_overall[original_metric_key] = new_name
                        final_display_cols_overall.append(new_name)
                
                agg_df_display_overall.rename(columns=renamed_cols_overall, inplace=True)
                final_display_cols_overall = [col for col in final_display_cols_overall if col in agg_df_display_overall.columns]
                
                formatter_overall = {}
                for original_key in metrics_to_display_non_placeholder: 
                    renamed_key_val = renamed_cols_overall.get(original_key) 
                    if renamed_key_val and renamed_key_val in final_display_cols_overall and \
                       original_key in agg_df.columns and pd.api.types.is_numeric_dtype(agg_df[original_key]):
                        formatter_overall[renamed_key_val] = "{:.4f}" 
                
                st.dataframe(
                    agg_df_display_overall[final_display_cols_overall].style.format(formatter=formatter_overall, na_rep='NaN'),
                    use_container_width=True
                )
                st.markdown("---")

                # --- Interpretation Section (Moved Here) ---
                st.markdown("#### 🔍 Interpreting Your Aggregated Results (Experimental)")
                with st.expander("💡 Interpreting Your Aggregated Results (Experimental)", expanded=False):
                    st.markdown("""
                    This section offers a general interpretation of the aggregated scores. Remember that these are heuristic-based and should be combined with a qualitative review of individual responses for a complete understanding.
                    Low scores don't always mean a "bad" model; they indicate areas where the model's output differs from the reference or desired behavior according to the specific metric.
                    """)
                    if agg_df is not None and not agg_df.empty:
                        for task_type_interp in agg_df['task_type'].unique():
                            st.markdown(f"#### Task: {task_type_interp}")
                            task_data_interp = agg_df[agg_df['task_type'] == task_type_interp]
                            for model_name_interp in task_data_interp['model'].unique():
                                model_scores_interp = task_data_interp[task_data_interp['model'] == model_name_interp].iloc[0]
                                st.markdown(f"**Model: `{model_name_interp}`**")
                                interpretations = []
                                suggestions = []

                                # Fluency & Similarity (BLEU, ROUGE, METEOR)
                                fluency_scores = {m: model_scores_interp.get(m) for m in ['bleu', 'rouge_l', 'meteor'] if m in model_scores_interp and pd.notna(model_scores_interp.get(m)) and not is_placeholder_metric(m)}
                                if fluency_scores:
                                    valid_fluency_scores = [s for s in fluency_scores.values() if pd.notna(s)]
                                    if valid_fluency_scores:
                                        avg_fluency = np.mean(valid_fluency_scores)
                                        if avg_fluency > 0.5: interpretations.append(f"✅ Generally good fluency and similarity to references (Avg. relevant score: {avg_fluency:.2f}).")
                                        elif avg_fluency > 0.2: interpretations.append(f"⚠️ Moderate fluency/similarity. Responses may differ noticeably from references (Avg. relevant score: {avg_fluency:.2f}).")
                                        else: interpretations.append(f"❌ Low fluency/similarity. Responses might be quite different from references or have linguistic issues (Avg. relevant score: {avg_fluency:.2f})."); suggestions.append("Review responses for clarity, grammar, and relevance. Consider prompt adjustments or fine-tuning on target-style data.")
                                
                                fact_presence = model_scores_interp.get('fact_presence_score')
                                if pd.notna(fact_presence) and not is_placeholder_metric('fact_presence_score'):
                                    if fact_presence > 0.7: interpretations.append(f"✅ Good inclusion of specified facts ({fact_presence:.2f}).")
                                    elif fact_presence > 0.4: interpretations.append(f"⚠️ Moderate inclusion of facts ({fact_presence:.2f}). Some key information might be missing.")
                                    else: interpretations.append(f"❌ Low inclusion of specified facts ({fact_presence:.2f})."); suggestions.append("Ensure `ref_facts` are accurate and present in good answers. For RAG, check context relevance and model's ability to extract from it.")
                                
                                completeness = model_scores_interp.get('completeness_score')
                                if pd.notna(completeness) and not is_placeholder_metric('completeness_score'):
                                    if completeness > 0.7: interpretations.append(f"✅ Good coverage of key points ({completeness:.2f}).")
                                    elif completeness > 0.4: interpretations.append(f"⚠️ Moderate coverage of key points ({completeness:.2f}). May not address all aspects.")
                                    else: interpretations.append(f"❌ Low coverage of key points ({completeness:.2f})."); suggestions.append("Ensure `ref_key_points` are well-defined. Model might need prompting to be more comprehensive or context might be lacking.")
                                
                                f1 = model_scores_interp.get('f1_score'); acc = model_scores_interp.get('accuracy')
                                if pd.notna(f1) and not is_placeholder_metric('f1_score'):
                                    if f1 > 0.75: interpretations.append(f"✅ Good classification performance (F1: {f1:.2f}).")
                                    elif f1 > 0.5: interpretations.append(f"⚠️ Moderate classification performance (F1: {f1:.2f}).")
                                    else: interpretations.append(f"❌ Low classification performance (F1: {f1:.2f})."); suggestions.append("Review misclassified examples. Consider more/better training data, feature engineering, or model architecture for classification tasks.")
                                elif pd.notna(acc) and not is_placeholder_metric('accuracy'): 
                                     if acc > 0.75: interpretations.append(f"✅ Good classification accuracy ({acc:.2f}).")
                                     else: interpretations.append(f"⚠️ Classification accuracy is {acc:.2f}. Consider checking precision/recall if available."); suggestions.append("Review misclassified examples. More/better training data might be needed.")
                                
                                length_ratio = model_scores_interp.get('length_ratio')
                                if pd.notna(length_ratio) and not is_placeholder_metric('length_ratio'):
                                    if 0.75 <= length_ratio <= 1.25: interpretations.append(f"✅ Good response length relative to reference (Ratio: {length_ratio:.2f}).")
                                    elif length_ratio < 0.5: interpretations.append(f"⚠️ Responses may be too short (Ratio: {length_ratio:.2f})."); suggestions.append("Check if model is truncating answers or being overly brief. Adjust max tokens or prompt for more detail if needed.")
                                    elif length_ratio > 1.75: interpretations.append(f"⚠️ Responses may be too verbose (Ratio: {length_ratio:.2f})."); suggestions.append("Model might be adding unnecessary information. Prompt for conciseness or set stricter length limits.")
                                    else: interpretations.append(f"ℹ️ Length ratio is {length_ratio:.2f}. Assess if appropriate for the task.")
                                
                                safety_score = model_scores_interp.get('safety_keyword_score')
                                if pd.notna(safety_score) and not is_placeholder_metric('safety_keyword_score'):
                                    if safety_score < 1.0: interpretations.append(f"🚨 Safety alert! Basic keyword check failed for some responses (Score: {safety_score:.2f}). MANUAL REVIEW OF INDIVIDUAL CASES IS CRITICAL."); suggestions.append("Implement stricter content filtering or moderation. Review prompts that might lead to unsafe content.")
                                    else: interpretations.append("✅ Basic safety keyword check passed.")
                                
                                pii_score = model_scores_interp.get('pii_detection_score')
                                if pd.notna(pii_score) and not is_placeholder_metric('pii_detection_score'):
                                    if pii_score < 1.0: interpretations.append(f"🚨 Privacy alert! Basic PII pattern check failed for some responses (Score: {pii_score:.2f}). MANUAL REVIEW IS CRITICAL."); suggestions.append("Enhance PII detection and scrubbing. Review data handling policies and prompts.")
                                    else: interpretations.append("✅ Basic PII detection check passed.")

                                if interpretations:
                                    st.markdown("**Observations:**")
                                    for o_item in interpretations: st.markdown(f"- {o_item}") 
                                if suggestions:
                                    st.markdown("**Potential Actions:**")
                                    for s_item in suggestions: st.markdown(f"- {s_item}") 
                                if not interpretations and not suggestions:
                                    st.markdown("_No specific interpretations generated based on available non-placeholder scores for this model/task combination._")
                                st.markdown("---") 
                    else: st.info("Run an evaluation to see interpretations.")
                st.markdown("---") 

                st.markdown("#### 📊 Task Specific Metric Table & Chart 📈  ")
                available_tasks_agg = sorted(agg_df['task_type'].unique()) if 'task_type' in agg_df else []
                if not available_tasks_agg: st.info("No tasks found in aggregated results.")
                else:
                    task_tabs_agg = st.tabs([f"Task: {task}" for task in available_tasks_agg])
                    for i, task_type in enumerate(available_tasks_agg):
                        with task_tabs_agg[i]:
                            task_df_agg = agg_df[agg_df['task_type'] == task_type].copy()
                            task_specific_metrics_for_task_dim_view = [
                                m for m in get_metrics_for_task(task_type) 
                                if m in agg_df.columns and m in metrics_to_display_non_placeholder 
                            ]
                            if not task_specific_metrics_for_task_dim_view:
                                st.info(f"No relevant, non-placeholder metrics to display for task '{task_type}'.")
                                continue

                            relevant_categories_agg = sorted(list(set(METRIC_INFO[m]['category'] for m in task_specific_metrics_for_task_dim_view if m in METRIC_INFO)))
                            ordered_relevant_categories_agg = [cat for cat in CATEGORY_ORDER if cat in relevant_categories_agg]

                            if not ordered_relevant_categories_agg: st.info(f"No metric categories for task '{task_type}'.")
                            else:
                                dimension_tabs_agg = st.tabs([f"{cat}" for cat in ordered_relevant_categories_agg])
                                for j_dim, category in enumerate(ordered_relevant_categories_agg):
                                    with dimension_tabs_agg[j_dim]:
                                        metrics_in_category_task_agg = [m for m in task_specific_metrics_for_task_dim_view if METRIC_INFO.get(m, {}).get('category') == category]
                                        if not metrics_in_category_task_agg : 
                                            st.write(f"_No '{category}' metrics available or selected for display in this task._")
                                            continue
                                        
                                        cols_to_show_agg_dim = ['model', 'num_samples'] + metrics_in_category_task_agg
                                        cols_to_show_present_agg_dim = [c for c in cols_to_show_agg_dim if c in task_df_agg.columns]
                                        
                                        st.markdown(f"###### {category} Metrics Table (Aggregated for Task: {task_type})")
                                        filtered_df_dim_agg = task_df_agg[cols_to_show_present_agg_dim].copy()
                                        new_cat_columns_agg = {}
                                        for col_key_dim in filtered_df_dim_agg.columns:
                                            if col_key_dim in metrics_in_category_task_agg:
                                                indicator = get_metric_indicator(col_key_dim)
                                                col_title = get_metric_display_name(col_key_dim) 
                                                new_cat_columns_agg[col_key_dim] = f"{col_title} {indicator}".strip()
                                            elif col_key_dim in ['model', 'num_samples']:
                                                new_cat_columns_agg[col_key_dim] = col_key_dim.replace('_', ' ').title()
                                        filtered_df_dim_agg.rename(columns=new_cat_columns_agg, inplace=True)
                                        display_dim_cols_agg = [new_cat_columns_agg.get(col,col) for col in cols_to_show_present_agg_dim if new_cat_columns_agg.get(col,col) in filtered_df_dim_agg.columns]
                                        
                                        st.dataframe(
                                            filtered_df_dim_agg[display_dim_cols_agg].style.pipe(apply_color_gradient, METRIC_INFO), 
                                            use_container_width=True
                                        )

                                        st.markdown(f"###### {category} Charts (Aggregated for Task: {task_type})")
                                        plottable_metrics_agg = [m for m in metrics_in_category_task_agg if pd.api.types.is_numeric_dtype(task_df_agg[m])]
                                        if not plottable_metrics_agg: st.info("No numeric metrics for charting.")
                                        else:
                                            metric_display_options_agg = {f"{get_metric_display_name(m)} {get_metric_indicator(m)}".strip(): m for m in plottable_metrics_agg}
                                            selected_metric_display_agg = st.selectbox(
                                                f"Metric for {task_type} - {category}:", list(metric_display_options_agg.keys()),
                                                key=f"chart_sel_agg_{task_type}_{category.replace(' ','_')}_{j_dim}"
                                            )
                                            if selected_metric_display_agg:
                                                selected_metric_chart_agg = metric_display_options_agg[selected_metric_display_agg]
                                                metric_explanation_agg = METRIC_INFO.get(selected_metric_chart_agg, {}).get('explanation', "N/A")
                                                st.caption(f"**Definition ({get_metric_display_name(selected_metric_chart_agg)}):** {metric_explanation_agg}")
                                                try:
                                                    fig_agg = px.bar(task_df_agg, x='model', y=selected_metric_chart_agg, title=f"{selected_metric_display_agg} Scores",
                                                                labels={'model': 'Model / Config', selected_metric_chart_agg: selected_metric_display_agg},
                                                                color='model', text_auto='.4f')
                                                    fig_agg.update_layout(xaxis_title="Model / Config", yaxis_title=selected_metric_display_agg); fig_agg.update_traces(textposition='outside')
                                                    st.plotly_chart(fig_agg, use_container_width=True)
                                                except Exception as e_chart: st.error(f"Chart error: {e_chart}")
            st.divider()
            st.subheader("Download Aggregated Reports")
            if agg_df is not None and not agg_df.empty:
                col1_agg_dl, col2_agg_dl = st.columns(2)
                csv_data_agg = agg_df.to_csv(index=False).encode('utf-8') 
                md_content_agg = f"# LLM Evaluation Aggregated Report ({datetime.datetime.now():%Y-%m-%d %H:%M})\n\n"
                agg_df_md_display_dl = agg_df.copy() 
                agg_df_md_display_dl.rename(columns=renamed_cols_overall, inplace=True) 
                md_content_agg += agg_df_md_display_dl[final_display_cols_overall].to_markdown(index=False, floatfmt=".4f")
                md_content_agg += "\n\n---\n_End of Aggregated Summary_"
                with col1_agg_dl: st.download_button("⬇️ CSV Aggregated Results", csv_data_agg, f"aggregated_eval_results_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv", key="dl_csv_agg")
                with col2_agg_dl: st.download_button("⬇️ MD Aggregated Summary", md_content_agg.encode('utf-8'), f"aggregated_eval_summary_{datetime.datetime.now():%Y%m%d_%H%M%S}.md", "text/markdown", key="dl_md_agg")
        else: st.info("No aggregated results to display. Run an evaluation.")

    with res_tab_ind:
        st.subheader("📊 Individual Test Case Scores")
        if st.session_state.individual_scores_df is not None and not st.session_state.individual_scores_df.empty:
            ind_df_display = st.session_state.individual_scores_df.copy()

            renamed_cols_ind_display = {}
            original_metric_keys_in_ind_df = [col for col in ind_df_display.columns if col in METRIC_INFO]
            for m_key in original_metric_keys_in_ind_df:
                renamed_cols_ind_display[m_key] = f"{get_metric_display_name(m_key, include_placeholder_tag=True)} {get_metric_indicator(m_key)}".strip()
            ind_df_display.rename(columns=renamed_cols_ind_display, inplace=True)

            id_cols = ['id', 'task_type', 'model', 'test_description']
            metric_cols_display_names = [renamed_cols_ind_display.get(m_key, m_key) for m_key in original_metric_keys_in_ind_df if renamed_cols_ind_display.get(m_key, m_key) in ind_df_display.columns]
            interpretation_cols = ['Observations', 'Potential Actions'] 
            input_output_cols = ['question', 'contexts', 'ground_truth', 'answer', 'ref_facts', 'ref_key_points']
            
            final_order_ind_display = []
            for col in id_cols:
                if col in ind_df_display.columns: final_order_ind_display.append(col)
            final_order_ind_display.extend(metric_cols_display_names) 
            for col in interpretation_cols: 
                if col in ind_df_display.columns: final_order_ind_display.append(col)
            for col in input_output_cols:
                if col in ind_df_display.columns: final_order_ind_display.append(col)
            remaining_other_cols_ind = sorted([col for col in ind_df_display.columns if col not in final_order_ind_display and not col.startswith('_st_')])
            final_order_ind_display.extend(remaining_other_cols_ind)
            final_order_ind_display = [col for col in final_order_ind_display if col in ind_df_display.columns]

            st.info("Displaying all scores and interpretations for each test case. Use column headers to sort. Download full table below.")
            st.dataframe(ind_df_display[final_order_ind_display].style.pipe(apply_color_gradient, METRIC_INFO), use_container_width=True)
            
            # --- New: Interpretation for Selected Individual Test Case ---
            st.divider()
            st.subheader("🔍 Detailed Interpretation for a Single Test Case")
            if 'id' in st.session_state.individual_scores_df.columns:
                # Ensure 'id' column is string for consistent matching with selectbox options
                available_ids = st.session_state.individual_scores_df['id'].astype(str).unique().tolist()
                if not available_ids:
                    st.warning("No test case IDs found in the individual results.")
                else:
                    # Add a "None" option to allow deselecting or showing no interpretation initially
                    options_for_selectbox = ["<Select a Test Case ID>"] + available_ids
                    selected_id_for_interp = st.selectbox(
                        "Select Test Case ID to see detailed interpretation:",
                        options=options_for_selectbox,
                        index=0, # Default to "<Select...>"
                        key="individual_case_interp_selector"
                    )

                    if selected_id_for_interp and selected_id_for_interp != "<Select a Test Case ID>":
                        selected_case_data = st.session_state.individual_scores_df[
                            st.session_state.individual_scores_df['id'].astype(str) == selected_id_for_interp
                        ]
                        if not selected_case_data.empty:
                            case_to_show = selected_case_data.iloc[0]
                            st.markdown(f"**Test Case ID:** `{case_to_show.get('id', 'N/A')}`")
                            st.markdown(f"**Model:** `{case_to_show.get('model', 'N/A')}`")
                            st.markdown(f"**Task Type:** `{case_to_show.get('task_type', 'N/A')}`")
                            if pd.notna(case_to_show.get('test_description')):
                                st.markdown(f"**Description:** {case_to_show.get('test_description')}")

                            st.markdown("**Observations:**")
                            if pd.notna(case_to_show.get('Observations')) and case_to_show.get('Observations').strip():
                                st.markdown(case_to_show.get('Observations'))
                            else:
                                st.markdown("_No specific observations generated for this case._")

                            st.markdown("**Potential Actions:**")
                            if pd.notna(case_to_show.get('Potential Actions')) and case_to_show.get('Potential Actions').strip():
                                st.markdown(case_to_show.get('Potential Actions'))
                            else:
                                st.markdown("_No specific automated suggestions for this case. Review manually if scores were low._")
                            
                            # Optionally, show the Q, A, GT for context
                            with st.expander("View Question, Ground Truth, and Answer for this case"):
                                st.markdown(f"**Question:**\n```\n{case_to_show.get('question', '')}\n```")
                                st.markdown(f"**Ground Truth:**\n```\n{case_to_show.get('ground_truth', '')}\n```")
                                st.markdown(f"**LLM Answer:**\n```\n{case_to_show.get('answer', '')}\n```")
                                if pd.notna(case_to_show.get('contexts')):
                                     st.markdown(f"**Contexts:**\n```\n{case_to_show.get('contexts', '')}\n```")


                        else:
                            st.warning(f"Could not find data for selected ID: {selected_id_for_interp}")
            else:
                st.info("Run evaluation to generate individual scores with IDs for detailed interpretation.")


            st.divider(); st.subheader("Download Individual Scores Report (with Interpretations)")
            csv_download_df_ind = st.session_state.individual_scores_df.copy() 
            
            csv_id_cols = ['id', 'task_type', 'model', 'test_description']
            csv_metric_cols = [m_key for m_key in METRIC_INFO.keys() if m_key in csv_download_df_ind.columns] 
            csv_interp_cols = ['Observations', 'Potential Actions']
            csv_input_output_cols = ['question', 'contexts', 'ground_truth', 'answer', 'ref_facts', 'ref_key_points']

            csv_final_order = []
            for col_group in [csv_id_cols, csv_metric_cols, csv_interp_cols, csv_input_output_cols]:
                for col in col_group:
                    if col in csv_download_df_ind.columns: csv_final_order.append(col)
            
            csv_remaining_cols = sorted([col for col in csv_download_df_ind.columns if col not in csv_final_order and not col.startswith('_st_')])
            csv_final_order.extend(csv_remaining_cols)
            csv_final_order = [col for col in csv_final_order if col in csv_download_df_ind.columns] 

            csv_data_ind = csv_download_df_ind[csv_final_order].to_csv(index=False, float_format="%.4f").encode('utf-8')
            st.download_button("⬇️ CSV Individual Scores & Interpretations", csv_data_ind, f"individual_eval_scores_interpreted_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv", key="dl_csv_ind_interpreted")
        else: st.info("No individual scores to display. Run an evaluation.")

with tab_data_editor:
    st.header("Manage Evaluation Data")
    st.markdown("Manually add new evaluation rows or edit existing data. Data loaded/generated via sidebar appears here.")
    st.subheader("Add New Evaluation Row")
    with st.form("add_case_form", clear_on_submit=True):
        col_form_id, col_form_task, col_form_model = st.columns(3)
        with col_form_id: add_id = st.text_input("Test Case ID (Optional)", key="add_id_input", placeholder="e.g., rag_case_001")
        with col_form_task: add_task_type = st.selectbox("Task Type*", list(get_supported_tasks()), key="add_task_type_select", index=None, placeholder="Select Task...")
        with col_form_model: add_model = st.text_input("LLM/Model Config*", key="add_model_input", placeholder="e.g., MyModel_v1.2_temp0.7")
        add_description = st.text_input("Test Description (Optional)", key="add_description_input", placeholder="Briefly describe this test case's purpose...")
        add_question = st.text_area("Question / Input Text*", key="add_question_input", placeholder="Input query, text to summarize/classify, or chatbot utterance.", height=100)
        add_ground_truth = st.text_area("Ground Truth / Reference*", key="add_ground_truth_input", placeholder="Ideal answer, reference summary, correct label, or reference response.", height=100)
        add_answer = st.text_area("LLM's Actual Answer / Prediction*", key="add_answer_input", placeholder="Actual output generated by the LLM.", height=100)
        col_form_context, col_form_facts, col_form_kps = st.columns(3)
        with col_form_context: add_contexts = st.text_area("Contexts (Optional)", key="add_contexts_input", placeholder="For RAG, retrieved context.", height=75)
        with col_form_facts: add_ref_facts = st.text_input("Reference Facts (Optional, comma-separated)", key="add_ref_facts_input", placeholder="fact A,fact B")
        with col_form_kps: add_ref_key_points = st.text_input("Reference Key Points (Optional, comma-separated)", key="add_ref_key_points_input", placeholder="point 1,point 2")
        submitted_add_row = st.form_submit_button("➕ Add Evaluation Row to Editor")
        if submitted_add_row:
            if not all([st.session_state.add_task_type_select, st.session_state.add_model_input, st.session_state.add_question_input, st.session_state.add_ground_truth_input, st.session_state.add_answer_input]):
                st.error("Required fields (*) missing.")
            else:
                current_df = st.session_state.edited_test_cases_df
                if add_id and not current_df.empty and 'id' in current_df.columns and add_id in current_df['id'].astype(str).values:
                    st.error(f"ID '{add_id}' exists. Use unique ID or leave blank.")
                else:
                    new_row_dict = {'id': add_id if add_id else f"manual_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}",
                                   'task_type': st.session_state.add_task_type_select, 'model': st.session_state.add_model_input,
                                   'test_description': st.session_state.add_description_input or None, 'question': st.session_state.add_question_input,
                                   'contexts': st.session_state.add_contexts_input or None, 'ground_truth': st.session_state.add_ground_truth_input,
                                   'answer': st.session_state.add_answer_input, 'ref_facts': st.session_state.add_ref_facts_input or None,
                                   'ref_key_points': st.session_state.add_ref_key_points_input or None}
                    new_row_df = pd.DataFrame([new_row_dict])
                    if st.session_state.edited_test_cases_df.empty: st.session_state.edited_test_cases_df = new_row_df.fillna('')
                    else:
                        for col in new_row_df.columns: 
                            if col not in st.session_state.edited_test_cases_df.columns: st.session_state.edited_test_cases_df[col] = np.nan
                        for col in st.session_state.edited_test_cases_df.columns: 
                             if col not in new_row_df.columns: new_row_df[col] = np.nan
                        st.session_state.edited_test_cases_df = pd.concat([st.session_state.edited_test_cases_df, new_row_df], ignore_index=True).fillna('')
                    st.success(f"Row '{new_row_dict['id']}' added."); st.rerun() 

    st.divider(); st.subheader("Data Editor")
    if isinstance(st.session_state.edited_test_cases_df, pd.DataFrame) and not st.session_state.edited_test_cases_df.empty:
        st.markdown("Edit data below. Changes are used when 'Run Evaluation' is clicked. Add/delete rows as needed.")
        editor_col_order = ['id', 'task_type', 'model', 'test_description', 'question', 'ground_truth', 'answer', 'contexts', 'ref_facts', 'ref_key_points']
        available_cols_for_editor = [col for col in editor_col_order if col in st.session_state.edited_test_cases_df.columns]
        remaining_cols_for_editor = sorted([col for col in st.session_state.edited_test_cases_df.columns if col not in available_cols_for_editor])
        final_editor_cols = available_cols_for_editor + remaining_cols_for_editor
        edited_df_from_editor = st.data_editor(st.session_state.edited_test_cases_df[final_editor_cols].fillna(''),
                                               num_rows="dynamic", use_container_width=True, key="data_editor_main")
        st.session_state.edited_test_cases_df = edited_df_from_editor.copy()
        if not st.session_state.edited_test_cases_df.empty:
             csv_edited_data = st.session_state.edited_test_cases_df.fillna('').to_csv(index=False).encode('utf-8')
             st.download_button("⬇️ Download Edited Data (CSV)", csv_edited_data, f"edited_eval_cases_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv", key="dl_edited_data_csv")
    else: st.info("No data loaded. Use sidebar to load/generate or add rows using the form.")

with tab_format_guide:
    st.header("Input Data Format Guide (Flat Format)")
    st.markdown("""The framework expects input data (JSON, CSV, or Excel) in a **flat format**. Each row represents a single evaluation instance.""")
    guide_order = ['id', 'task_type', 'model', 'test_description', 'question', 'ground_truth', 'answer', 'contexts', 'ref_facts', 'ref_key_points']
    data_format_info = {
        "id": "**(Optional but Recommended)** Unique identifier for the row.",
        "task_type": "**(Required)** Task type (e.g., `rag_faq`, `summarization`).",
        "model": "**(Required)** LLM identifier (e.g., `ModelX_v1`, `GPT-3.5_temp0.7`).",
        "test_description": "**(Optional)** Brief description of the test case's purpose.",
        "question": "**(Required)** Input text/query for the LLM.",
        "ground_truth": "**(Required)** Reference answer, label, or summary.",
        "answer": "**(Required)** Actual output from the LLM.",
        "contexts": "**(Optional)** Context provided to LLM for RAG tasks.",
        "ref_facts": "**(Optional)** Comma-separated factual statements for `FactPresenceMetric`.",
        "ref_key_points": "**(Optional)** Comma-separated key points for `ChecklistCompletenessMetric`."}
    for col in guide_order:
        if col in data_format_info: st.markdown(f"- **`{col}`**: {data_format_info[col]}")
    st.subheader("Example Rows (Conceptual CSV/Excel Structure):")
    example_data = [{'id': 'rag_001', 'task_type': 'rag_faq', 'model': 'ModelAlpha', 'test_description': 'Capital of France', 'question': 'What is the capital of France?', 'contexts': 'Paris is the capital...', 'ground_truth': 'Paris is the capital.', 'answer': 'The capital is Paris.', 'ref_facts': 'Paris is capital', 'ref_key_points': 'Capital City'},
                    {'id': 'sum_001', 'task_type': 'summarization', 'model': 'ModelBeta', 'test_description': 'AI Summary', 'question': 'Summarize AI history.', 'contexts': '', 'ground_truth': 'AI evolved...', 'answer': 'AI started old, now ML.', 'ref_facts': '', 'ref_key_points': 'Evolution,ML'}]
    st.dataframe(pd.DataFrame(example_data).fillna(''))

with tab_metrics_tutorial:
    st.header("Metrics Tutorial & Explanations")
    st.markdown("Understand the metrics used. Metrics are grouped by evaluation dimension.")
    for category in CATEGORY_ORDER:
        with st.expander(f"Dimension: **{category}**", expanded=(category == CAT_TRUST)):
            st.markdown(f"*{DIMENSION_DESCRIPTIONS.get(category, '')}*"); st.markdown("---")
            metrics_in_this_category = METRICS_BY_CATEGORY.get(category, [])
            if not metrics_in_this_category: st.markdown("_No metrics in this dimension._")
            else:
                for metric_key in metrics_in_this_category:
                    info = METRIC_INFO.get(metric_key)
                    if info:
                        indicator = get_metric_indicator(metric_key)
                        # Use get_metric_display_name to include (Placeholder) tag
                        st.markdown(f"##### {get_metric_display_name(metric_key)} (`{metric_key}`) {indicator}")
                        st.markdown(f"**Use Case & Interpretation:** {info['explanation']}")
                        relevant_tasks = [task_name for task_name in get_supported_tasks() if metric_key in get_metrics_for_task(task_name)]
                        if relevant_tasks: st.markdown(f"**Commonly Used For Tasks:** `{'`, `'.join(relevant_tasks)}`")
                        else: st.markdown("**Commonly Used For Tasks:** (Specialized or placeholder).")
                        st.markdown("---")