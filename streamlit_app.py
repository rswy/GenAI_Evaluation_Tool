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
    from file_converter import convert_excel_to_data, convert_csv_to_data # FIELD_STRUCTURE_MAP not directly used in Streamlit for flat format
    from mock_data_generator import generate_mock_data_flat, save_mock_data
    from tasks.task_registry import get_metrics_for_task, get_supported_tasks, RAG_FAQ, SUMMARIZATION, CLASSIFICATION, CHATBOT
except ImportError as e:
    st.error(f"Framework Import Error: {e}. Please ensure all necessary files are in the 'src' directory and Python environment is set up correctly.")
    st.error(f"Current sys.path: {sys.path}")
    st.error(f"Project root evaluated as: {project_root}")
    st.stop()

# --- Metric Information (for display purposes) ---
# (Copied from original, ensure this matches your task_registry.py and metric outputs)
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

def get_metric_indicator(metric_key):
    info = METRIC_INFO.get(metric_key); return "⬆️" if info and info["higher_is_better"] else ("⬇️" if info else "")

def apply_color_gradient(styler, metric_info_dict_local):
    cmap_good = matplotlib.colormaps['RdYlGn']; cmap_bad = matplotlib.colormaps['RdYlGn_r']
    metric_cols_in_styler = [col for col in styler.columns if isinstance(col, str) and col.split(" ")[0].lower().replace(" ","_") in metric_info_dict_local]
    for col_name_with_indicator in metric_cols_in_styler:
        parts = col_name_with_indicator.split(" "); indicator = parts[-1] if parts else ""
        metric_name_parts = parts[:-1] if indicator in ["⬆️", "⬇️"] else parts
        metric_key = "_".join(metric_name_parts).lower()
        info = metric_info_dict_local.get(metric_key)
        if info and pd.api.types.is_numeric_dtype(styler.data[col_name_with_indicator]):
            cmap_to_use = cmap_good if info['higher_is_better'] else cmap_bad
            try: styler.background_gradient(cmap=cmap_to_use, subset=[col_name_with_indicator], axis=0, vmin=0.0, vmax=1.0)
            except Exception as e: warnings.warn(f"Style error for '{col_name_with_indicator}': {e}", RuntimeWarning)
    
    float_cols = styler.data.select_dtypes(include=['float', 'float64', 'float32']).columns
    format_dict = {col: '{:.4f}' for col in float_cols if col in styler.columns}
    styler.format(format_dict)
    return styler

def unflatten_df_to_test_cases(df):
    test_cases_list = []
    if df is None or df.empty: return []
    
    # Define expected columns for a flat test case
    # These are keys that should exist directly in each row of the DataFrame if it's flat
    direct_keys = ['id', 'task_type', 'model', 'question', 'contexts', 'ground_truth', 'answer', 'ref_facts', 'ref_key_points', 'test_description']

    for _, row_series in df.iterrows():
        row = row_series.to_dict()
        case = {}
        valid_case = True
        # Check for required fields in the flat format
        if pd.isna(row.get('task_type')) or pd.isna(row.get('model')) or \
           pd.isna(row.get('question')) or pd.isna(row.get('ground_truth')) or \
           pd.isna(row.get('answer')):
            st.warning(f"Skipping row due to missing required field(s) (task_type, model, question, ground_truth, answer): {row.get('id', 'Unknown ID')}")
            valid_case = False
            continue

        if valid_case:
            for key in direct_keys:
                if key in row and pd.notna(row[key]):
                    case[key] = str(row[key]) # Ensure string conversion for consistency
                else:
                    case[key] = None # Or empty string "" if preferred for optional fields

            # Add any other columns from the DataFrame row directly
            for col_name, value in row.items():
                if col_name not in case: # Avoid overwriting already processed direct keys
                    case[col_name] = str(value) if pd.notna(value) else None
            
            test_cases_list.append(case)
            
    return test_cases_list


# --- Streamlit App Configuration ---
st.set_page_config(layout="wide", page_title="LLM Evaluation Framework")
st.title("📊 LLM Evaluation Framework")
st.markdown("Evaluate LLM performance using pre-generated responses. This tool now supports both **aggregated summaries** and **individual test case scores**.")

# --- State Management ---
default_state_keys = {
    'test_cases_list_loaded': None, # Raw list of dicts from file/mockgen
    'edited_test_cases_df': pd.DataFrame(), # DataFrame for st.data_editor
    'aggregated_results_df': None, # Aggregated scores DataFrame
    'individual_scores_df': None,  # Individual scores DataFrame
    'data_source_info': None,
    'last_uploaded_file_name': None,
    'metrics_for_agg_display': [] # Metrics to show in aggregated views
}
for key, default_value in default_state_keys.items():
    if key not in st.session_state:
        st.session_state[key] = copy.deepcopy(default_value) # Use deepcopy for mutable defaults

# --- Sidebar ---
st.sidebar.header("⚙️ Input Options")
def clear_app_state():
    st.session_state.test_cases_list_loaded = None
    st.session_state.edited_test_cases_df = pd.DataFrame()
    st.session_state.aggregated_results_df = None
    st.session_state.individual_scores_df = None
    st.session_state.data_source_info = None
    # st.session_state.last_uploaded_file_name remains to prevent re-upload churn on widget interaction
    st.session_state.metrics_for_agg_display = []


input_method = st.sidebar.radio(
    "Choose data source:",
    ("Upload File", "Generate Mock Data"),
    key="input_method_radio",
    on_change=clear_app_state # Clear results if source type changes
)

if input_method == "Upload File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload (.xlsx, .csv, .json - Flat Format)",
        type=["xlsx", "csv", "json"],
        key="file_uploader" # No on_change here, process below
    )
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.last_uploaded_file_name:
            clear_app_state() # Clear previous results if a new file is chosen
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
                    if file_suffix == ".xlsx":
                        test_data_list_from_file = convert_excel_to_data(tmp_file_path)
                    elif file_suffix == ".csv":
                        test_data_list_from_file = convert_csv_to_data(tmp_file_path)
                    elif file_suffix == ".json":
                        test_data_list_from_file = load_data(tmp_file_path)
                
                if test_data_list_from_file:
                    st.session_state.test_cases_list_loaded = test_data_list_from_file
                    df_for_edit = pd.DataFrame(test_data_list_from_file)
                    # Ensure essential columns exist for the editor, even if None
                    required_cols_editor = ['id', 'task_type', 'model', 'question', 'contexts', 'ground_truth', 'answer', 'ref_facts', 'ref_key_points', 'test_description']
                    for col in required_cols_editor:
                        if col not in df_for_edit.columns:
                            df_for_edit[col] = None # Or np.nan, or ""
                    st.session_state.edited_test_cases_df = df_for_edit.copy().fillna('') # Fill NaN with empty for editor
                    st.session_state.data_source_info = f"Loaded {len(test_data_list_from_file)} rows from {uploaded_file.name} into editor."
                    st.sidebar.success(st.session_state.data_source_info)
                else:
                    if test_data_list_from_file == []: # Explicitly empty
                        raise ValueError("File loaded but was empty or contained no valid data.")
                    else: # None was returned by converter/loader
                        raise ValueError("Failed to load or convert data. Check file format and content. Converters might have printed more specific errors to console.")
            except Exception as e:
                st.session_state.data_source_info = f"Error processing {uploaded_file.name}: {e}"
                st.sidebar.error(st.session_state.data_source_info)
                clear_app_state() # Clear on error
            finally:
                if tmp_file_path and tmp_file_path.exists():
                    try: os.unlink(tmp_file_path)
                    except Exception as e_unlink: warnings.warn(f"Could not delete temp file {tmp_file_path}: {e_unlink}")
        # If uploaded_file is same as last_uploaded_file_name, do nothing to allow reruns without re-processing
elif input_method == "Generate Mock Data":
    st.sidebar.warning("Mock data provides example flat-format rows with sample facts/key points for testing evaluation dimensions.")
    if st.sidebar.button("Generate and Use Mock Data", key="generate_mock_button"):
        clear_app_state()
        try:
            with st.spinner("Generating mock evaluation data..."):
                mock_data_list = generate_mock_data_flat(num_samples_per_task=3) # Generates a list of dicts
            
            if mock_data_list:
                # Save mock data (optional, but good for transparency)
                # save_mock_data(mock_data_list, output_dir=data_dir, base_filename="streamlit_mock_data")
                
                st.session_state.test_cases_list_loaded = mock_data_list
                df_for_edit = pd.DataFrame(mock_data_list)
                required_cols_editor = ['id', 'task_type', 'model', 'question', 'contexts', 'ground_truth', 'answer', 'ref_facts', 'ref_key_points', 'test_description']
                for col in required_cols_editor:
                    if col not in df_for_edit.columns:
                        df_for_edit[col] = None
                st.session_state.edited_test_cases_df = df_for_edit.copy().fillna('') # Fill NaN for editor
                st.session_state.data_source_info = f"Using {len(mock_data_list)} generated mock rows, loaded into editor."
                st.sidebar.success(st.session_state.data_source_info)
            else:
                st.sidebar.error("Failed to generate mock data.")
        except Exception as e:
            st.sidebar.error(f"Error generating mock data: {e}")
            import traceback
            st.sidebar.text_area("Traceback", traceback.format_exc(), height=150)


# --- Main Content Area ---
if st.session_state.data_source_info:
    if "error" in st.session_state.data_source_info.lower() or "failed" in st.session_state.data_source_info.lower() :
        st.error(st.session_state.data_source_info)
    elif "loaded" in st.session_state.data_source_info.lower() or "using" in st.session_state.data_source_info.lower():
        st.success(st.session_state.data_source_info)
    else: # Processing, etc.
        st.info(st.session_state.data_source_info)

# --- Main Tabs ---
tab_titles_main = ["📊 Evaluation & Results", "📝 View/Edit/Add Data", "📄 Data Format Guide", "📖 Metrics Tutorial"]
tab_eval, tab_data_editor, tab_format_guide, tab_metrics_tutorial = st.tabs(tab_titles_main)

# --- Tab 1: Evaluation & Results ---
with tab_eval:
    st.header("Run Evaluation and View Results")
    run_button_disabled = not isinstance(st.session_state.edited_test_cases_df, pd.DataFrame) or st.session_state.edited_test_cases_df.empty
    
    if st.button("🚀 Run Evaluation on Data in Editor", disabled=run_button_disabled, key="run_eval_main_button", help="Evaluates the current data shown in the 'View/Edit/Add Data' tab."):
        if isinstance(st.session_state.edited_test_cases_df, pd.DataFrame) and not st.session_state.edited_test_cases_df.empty:
            st.session_state.aggregated_results_df = None # Clear previous aggregated results
            st.session_state.individual_scores_df = None  # Clear previous individual results
            st.session_state.metrics_for_agg_display = []

            with st.spinner("⏳ Evaluating... This may take a moment for larger datasets or complex metrics."):
                try:
                    # Convert the potentially edited DataFrame back to list of dicts for evaluator
                    # Ensure NaNs are handled correctly (e.g., converted to None or empty strings if appropriate for evaluator)
                    df_to_process = st.session_state.edited_test_cases_df.replace('', np.nan) # Replace empty strings with NaN for processing
                    test_cases_to_evaluate = unflatten_df_to_test_cases(df_to_process) # This function should handle the conversion

                    if not test_cases_to_evaluate:
                        raise ValueError("Data in editor is empty or could not be processed into valid test cases.")
                    
                    # Evaluator now returns two DataFrames
                    individual_df, aggregated_df = evaluate_model_responses(test_cases_to_evaluate)
                    
                    st.session_state.individual_scores_df = individual_df
                    st.session_state.aggregated_results_df = aggregated_df

                    if aggregated_df is not None and not aggregated_df.empty:
                        # Determine metrics to display in aggregated views (filter out all-zero/NaN cols)
                        metrics_present = [col for col in aggregated_df.columns if col in METRIC_INFO]
                        metrics_to_show_agg = []
                        for metric_col in metrics_present:
                            try:
                                if pd.api.types.is_numeric_dtype(aggregated_df[metric_col]):
                                    # Check if all values are NaN or close to zero
                                    is_all_zero_or_nan = (aggregated_df[metric_col].isna() | (aggregated_df[metric_col].abs() < 1e-9)).all()
                                    if not is_all_zero_or_nan:
                                        metrics_to_show_agg.append(metric_col)
                                else: # Keep non-numeric metric columns if any (shouldn't be for scores)
                                    metrics_to_show_agg.append(metric_col)
                            except KeyError:
                                warnings.warn(f"Metric column '{metric_col}' not found in aggregated results during filtering.")
                        st.session_state.metrics_for_agg_display = metrics_to_show_agg
                        st.success("✅ Evaluation complete! View aggregated and individual results below.")
                    elif individual_df is not None and not individual_df.empty:
                         st.warning("⚠️ Evaluation ran and produced individual scores, but aggregated results are empty. This might happen if no numeric metrics could be aggregated or grouping failed.")
                    else:
                        st.warning("⚠️ Evaluation finished, but no results were produced (both individual and aggregated are empty). Please check your input data and metric configurations.")
                except Exception as e:
                     st.error(f"An error occurred during evaluation: {e}")
                     import traceback
                     st.error(f"Traceback: {traceback.format_exc()}")
        else:
            st.warning("No data in the editor to evaluate. Please load or generate data first using the sidebar.")

    st.divider()
    st.header("Evaluation Results")

    # Create sub-tabs for Aggregated and Individual Results
    res_tab_agg, res_tab_ind = st.tabs(["📈 Aggregated Results", "📄 Individual Scores"])

    with res_tab_agg:
        st.subheader("Aggregated Scores per Task & Model")
        if st.session_state.aggregated_results_df is not None and not st.session_state.aggregated_results_df.empty:
            agg_df = st.session_state.aggregated_results_df
            metrics_to_display_agg = st.session_state.metrics_for_agg_display

            if not metrics_to_display_agg:
                st.info("No metrics to display in the aggregated summary (they might have been all zero or NaN and were filtered out, or no numeric metrics were computed). Displaying raw aggregated data if available.")
                st.dataframe(agg_df, use_container_width=True)
            else:
                st.markdown("##### Overall Summary Table (Aggregated)")
                agg_df_display = agg_df.copy()
                renamed_cols_agg_summary = {}
                for col in agg_df_display.columns:
                    if col in metrics_to_display_agg: # Only rename displayable metrics
                        indicator = get_metric_indicator(col)
                        col_title = METRIC_INFO.get(col, {}).get('name', col.replace('_', ' ').title())
                        renamed_cols_agg_summary[col] = f"{col_title} {indicator}".strip()
                    elif col in ['task_type', 'model', 'num_samples']:
                        renamed_cols_agg_summary[col] = col.replace('_', ' ').title()
                agg_df_display.rename(columns=renamed_cols_agg_summary, inplace=True)
                
                # Ensure all renamed columns actually exist before trying to display them
                display_cols_agg_summary = [renamed_cols_agg_summary.get(col, col) for col in agg_df.columns if col in renamed_cols_agg_summary or col in ['task_type', 'model', 'num_samples']]
                display_cols_agg_summary = [col for col in display_cols_agg_summary if col in agg_df_display.columns]


                st.dataframe(
                    agg_df_display[display_cols_agg_summary].style.pipe(apply_color_gradient, METRIC_INFO),
                    use_container_width=True
                )
                st.markdown("---")

                st.markdown("##### Aggregated Results by Task & Dimension")
                available_tasks_agg = sorted(agg_df['task_type'].unique()) if 'task_type' in agg_df else []
                if not available_tasks_agg:
                    st.info("No tasks found in aggregated results to display by dimension.")
                else:
                    task_tabs_agg = st.tabs([f"Task: {task}" for task in available_tasks_agg])
                    for i, task_type in enumerate(available_tasks_agg):
                        with task_tabs_agg[i]:
                            task_df_agg = agg_df[agg_df['task_type'] == task_type].copy()
                            task_metrics_calculated_agg = [col for col in metrics_to_display_agg if col in task_df_agg.columns and col in get_metrics_for_task(task_type)]
                            
                            if not task_metrics_calculated_agg:
                                st.info(f"No displayable aggregated metrics for task '{task_type}'.")
                                continue

                            relevant_categories_agg = sorted(list(set(METRIC_INFO[m]['category'] for m in task_metrics_calculated_agg if m in METRIC_INFO)))
                            ordered_relevant_categories_agg = [cat for cat in CATEGORY_ORDER if cat in relevant_categories_agg]

                            if not ordered_relevant_categories_agg:
                                st.info(f"No metric categories to display for task '{task_type}'.")
                                continue

                            dimension_tabs_agg = st.tabs([f"{cat}" for cat in ordered_relevant_categories_agg])
                            for j, category in enumerate(ordered_relevant_categories_agg):
                                with dimension_tabs_agg[j]:
                                    metrics_in_category_task_agg = [m for m in task_metrics_calculated_agg if METRIC_INFO.get(m, {}).get('category') == category]
                                    cols_to_show_agg_dim = ['model', 'num_samples'] + metrics_in_category_task_agg
                                    cols_to_show_present_agg_dim = [c for c in cols_to_show_agg_dim if c in task_df_agg.columns]

                                    if len(cols_to_show_present_agg_dim) <= 2: # Only model and num_samples
                                        st.write(f"_No '{category}' metrics available or selected for display in this task's aggregated results._")
                                        continue
                                    
                                    st.markdown(f"###### {category} Metrics Table (Aggregated)")
                                    filtered_df_dim_agg = task_df_agg[cols_to_show_present_agg_dim].copy()
                                    new_cat_columns_agg = {}
                                    for col in filtered_df_dim_agg.columns:
                                        if col in metrics_to_display_agg:
                                            indicator = get_metric_indicator(col)
                                            col_title = METRIC_INFO.get(col, {}).get('name', col.replace('_', ' ').title())
                                            new_cat_columns_agg[col] = f"{col_title} {indicator}".strip()
                                        elif col in ['model', 'num_samples']:
                                            new_cat_columns_agg[col] = col.replace('_', ' ').title()
                                    filtered_df_dim_agg.rename(columns=new_cat_columns_agg, inplace=True)
                                    display_dim_cols_agg = [new_cat_columns_agg.get(col,col) for col in cols_to_show_present_agg_dim]
                                    
                                    st.dataframe(
                                        filtered_df_dim_agg[display_dim_cols_agg].style.pipe(apply_color_gradient, METRIC_INFO),
                                        use_container_width=True
                                    )

                                    st.markdown(f"###### {category} Charts (Aggregated)")
                                    plottable_metrics_agg = [m for m in metrics_in_category_task_agg if pd.api.types.is_numeric_dtype(task_df_agg[m])]
                                    if not plottable_metrics_agg:
                                        st.info("No numeric metrics in this category for charting.")
                                        continue
                                    
                                    metric_display_options_agg = {f"{METRIC_INFO.get(m, {'name': m.replace('_',' ').title()})['name']} {get_metric_indicator(m)}".strip(): m for m in plottable_metrics_agg}
                                    selected_metric_display_agg = st.selectbox(
                                        f"Metric for {task_type} - {category}:",
                                        list(metric_display_options_agg.keys()),
                                        key=f"chart_sel_agg_{task_type}_{category.replace(' ','_')}_{j}"
                                    )
                                    
                                    if selected_metric_display_agg:
                                        selected_metric_chart_agg = metric_display_options_agg[selected_metric_display_agg]
                                        metric_explanation_agg = METRIC_INFO.get(selected_metric_chart_agg, {}).get('explanation', "No explanation available.")
                                        st.caption(f"**Definition ({METRIC_INFO.get(selected_metric_chart_agg,{}).get('name', selected_metric_chart_agg)}):** {metric_explanation_agg}")
                                        
                                        try:
                                            fig_agg = px.bar(task_df_agg, x='model', y=selected_metric_chart_agg,
                                                        title=f"{selected_metric_display_agg} Scores",
                                                        labels={'model': 'Model / Config', selected_metric_chart_agg: selected_metric_display_agg},
                                                        color='model', text_auto='.4f')
                                            fig_agg.update_layout(xaxis_title="Model / Config", yaxis_title=selected_metric_display_agg)
                                            fig_agg.update_traces(textposition='outside')
                                            st.plotly_chart(fig_agg, use_container_width=True)
                                        except Exception as e_chart:
                                            st.error(f"Could not generate chart: {e_chart}")
            # Download Buttons for aggregated results
            st.divider()
            st.subheader("Download Aggregated Reports")
            if agg_df is not None and not agg_df.empty:
                col1_agg_dl, col2_agg_dl = st.columns(2)
                csv_data_agg = agg_df.to_csv(index=False).encode('utf-8')
                
                # Markdown for aggregated
                md_content_agg = f"# LLM Evaluation Aggregated Report ({datetime.datetime.now():%Y-%m-%d %H:%M})\n\n"
                # Use the display-formatted DataFrame for Markdown
                md_content_agg += agg_df_display[display_cols_agg_summary].to_markdown(index=False, floatfmt=".4f") # Use the already renamed one
                md_content_agg += "\n\n---\n_End of Aggregated Summary_"

                with col1_agg_dl:
                    st.download_button("⬇️ CSV Aggregated Results", csv_data_agg, f"aggregated_eval_results_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv", key="dl_csv_agg")
                with col2_agg_dl:
                    st.download_button("⬇️ MD Aggregated Summary", md_content_agg.encode('utf-8'), f"aggregated_eval_summary_{datetime.datetime.now():%Y%m%d_%H%M%S}.md", "text/markdown", key="dl_md_agg")
        else:
            st.info("No aggregated results to display. Please run an evaluation.")

    with res_tab_ind:
        st.subheader("Individual Test Case Scores")
        if st.session_state.individual_scores_df is not None and not st.session_state.individual_scores_df.empty:
            ind_df = st.session_state.individual_scores_df
            
            # Reorder columns for better readability: original inputs first, then metrics
            # These are typical input columns
            original_input_cols = ['id', 'task_type', 'model', 'test_description', 'question', 'contexts', 'ground_truth', 'answer', 'ref_facts', 'ref_key_points']
            present_original_cols = [col for col in original_input_cols if col in ind_df.columns]
            
            # Metrics are any other columns not in the original input set (and not internal like _st_filter)
            # A more robust way to find metric columns is to check against METRIC_INFO keys
            calculated_metric_cols = sorted([col for col in ind_df.columns if col in METRIC_INFO and col not in present_original_cols])
            
            # Any other remaining columns (custom user columns, etc.)
            remaining_other_cols = sorted([col for col in ind_df.columns if col not in present_original_cols and col not in calculated_metric_cols and not col.startswith('_st_')])

            final_order_ind = present_original_cols + calculated_metric_cols + remaining_other_cols
            # Ensure all columns in final_order_ind actually exist in ind_df
            final_order_ind = [col for col in final_order_ind if col in ind_df.columns]

            st.info("Displaying all scores for each test case. Use the column headers to sort. You can download the full table below.")
            st.dataframe(ind_df[final_order_ind].style.pipe(apply_color_gradient, METRIC_INFO), use_container_width=True)

            # Download Button for individual results
            st.divider()
            st.subheader("Download Individual Scores Report")
            csv_data_ind = ind_df[final_order_ind].to_csv(index=False, float_format="%.4f").encode('utf-8')
            st.download_button("⬇️ CSV Individual Scores", csv_data_ind, f"individual_eval_scores_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv", key="dl_csv_ind")
        else:
            st.info("No individual scores to display. Please run an evaluation.")


# --- Tab 2: View/Edit/Add Data ---
with tab_data_editor:
    st.header("Manage Evaluation Data")
    st.markdown("Manually add new evaluation rows or edit existing data below. The data loaded or generated via the sidebar appears here for editing.")
    
    st.subheader("Add New Evaluation Row")
    with st.form("add_case_form", clear_on_submit=True):
        col_form_id, col_form_task, col_form_model = st.columns(3)
        with col_form_id:
            add_id = st.text_input("Test Case ID (Optional)", key="add_id_input", placeholder="e.g., rag_case_001")
        with col_form_task:
            add_task_type = st.selectbox("Task Type*", list(get_supported_tasks()), key="add_task_type_select", index=None, placeholder="Select Task...")
        with col_form_model:
            add_model = st.text_input("LLM/Model Config*", key="add_model_input", placeholder="e.g., MyModel_v1.2_temp0.7")

        add_description = st.text_input("Test Description (Optional)", key="add_description_input", placeholder="Briefly describe this test case's purpose...")
        
        add_question = st.text_area("Question / Input Text*", key="add_question_input", placeholder="The input query, text to summarize, text for classification, or chatbot utterance.", height=100)
        add_ground_truth = st.text_area("Ground Truth / Reference*", key="add_ground_truth_input", placeholder="The ideal answer, reference summary, correct label, or reference chatbot response.", height=100)
        add_answer = st.text_area("LLM's Actual Answer / Prediction*", key="add_answer_input", placeholder="The actual output generated by the LLM for the given input.", height=100)
        
        col_form_context, col_form_facts, col_form_kps = st.columns(3)
        with col_form_context:
            add_contexts = st.text_area("Contexts (Optional)", key="add_contexts_input", placeholder="For RAG, the retrieved context. Not directly used by all metrics.", height=75)
        with col_form_facts:
            add_ref_facts = st.text_input("Reference Facts (Optional, comma-separated)", key="add_ref_facts_input", placeholder="fact A,fact B,another fact")
        with col_form_kps:
            add_ref_key_points = st.text_input("Reference Key Points (Optional, comma-separated)", key="add_ref_key_points_input", placeholder="point 1,point 2,main idea")

        submitted_add_row = st.form_submit_button("➕ Add Evaluation Row to Editor")

        if submitted_add_row:
            if not st.session_state.add_task_type_select or \
               not st.session_state.add_model_input or \
               not st.session_state.add_question_input or \
               not st.session_state.add_ground_truth_input or \
               not st.session_state.add_answer_input:
                st.error("Please fill in all required fields marked with *.")
            else:
                # Check for duplicate ID if an ID is provided
                current_df = st.session_state.edited_test_cases_df
                if add_id and not current_df.empty and 'id' in current_df.columns and add_id in current_df['id'].astype(str).values:
                    st.error(f"Error: Test Case ID '{add_id}' already exists. Please use a unique ID or leave it blank for an auto-generated one.")
                else:
                    new_row_dict = {
                        'id': add_id if add_id else f"manual_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}",
                        'task_type': st.session_state.add_task_type_select,
                        'model': st.session_state.add_model_input,
                        'test_description': st.session_state.add_description_input if st.session_state.add_description_input else None,
                        'question': st.session_state.add_question_input,
                        'contexts': st.session_state.add_contexts_input if st.session_state.add_contexts_input else None,
                        'ground_truth': st.session_state.add_ground_truth_input,
                        'answer': st.session_state.add_answer_input,
                        'ref_facts': st.session_state.add_ref_facts_input if st.session_state.add_ref_facts_input else None,
                        'ref_key_points': st.session_state.add_ref_key_points_input if st.session_state.add_ref_key_points_input else None
                    }
                    new_row_df = pd.DataFrame([new_row_dict])
                    
                    if st.session_state.edited_test_cases_df.empty:
                        st.session_state.edited_test_cases_df = new_row_df.fillna('')
                    else:
                        # Align columns before concat
                        for col in new_row_df.columns:
                            if col not in st.session_state.edited_test_cases_df.columns:
                                st.session_state.edited_test_cases_df[col] = np.nan # Add new column if not exists
                        for col in st.session_state.edited_test_cases_df.columns:
                             if col not in new_row_df.columns:
                                 new_row_df[col] = np.nan


                        st.session_state.edited_test_cases_df = pd.concat([st.session_state.edited_test_cases_df, new_row_df], ignore_index=True).fillna('')
                    
                    st.success(f"Row '{new_row_dict['id']}' added to the editor below.")
                    # Form fields are cleared due to clear_on_submit=True and st.rerun() is implicitly handled by Streamlit forms.

    st.divider()
    st.subheader("Data Editor")
    if isinstance(st.session_state.edited_test_cases_df, pd.DataFrame) and not st.session_state.edited_test_cases_df.empty:
        st.markdown("Edit data directly in the table. Changes are used when 'Run Evaluation' is clicked. You can also add or delete rows.")
        
        # Define the desired column order for the editor (can be a subset or all)
        editor_col_order = ['id', 'task_type', 'model', 'test_description', 'question', 'ground_truth', 'answer', 'contexts', 'ref_facts', 'ref_key_points']
        # Filter to only columns present in the DataFrame, maintaining order
        available_cols_for_editor = [col for col in editor_col_order if col in st.session_state.edited_test_cases_df.columns]
        # Add any other columns not in the preferred order to the end
        remaining_cols_for_editor = sorted([col for col in st.session_state.edited_test_cases_df.columns if col not in available_cols_for_editor])
        final_editor_cols = available_cols_for_editor + remaining_cols_for_editor

        edited_df_from_editor = st.data_editor(
            st.session_state.edited_test_cases_df[final_editor_cols].fillna(''), # Work with a copy, ensure NaNs are empty strings for editor
            num_rows="dynamic", # Allows adding/deleting rows
            use_container_width=True,
            key="data_editor_main"
        )
        # Update session state with changes from the editor
        st.session_state.edited_test_cases_df = edited_df_from_editor.copy() # .replace('', np.nan) # Convert empty strings back to NaN if needed for processing

        if not st.session_state.edited_test_cases_df.empty:
             csv_edited_data = st.session_state.edited_test_cases_df.fillna('').to_csv(index=False).encode('utf-8')
             st.download_button("⬇️ Download Edited Data (CSV)", csv_edited_data, f"edited_eval_cases_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv", key="dl_edited_data_csv")
    else:
        st.info("No data loaded or generated. Use the sidebar to load a file or generate mock data. You can then add rows using the form above or edit here.")


# --- Tab 3: Data Format Guide ---
with tab_format_guide:
    st.header("Input Data Format Guide (Flat Format)")
    st.markdown("""
    The framework expects input data (JSON, CSV, or Excel) in a **flat format**. This means each row in your file represents a single evaluation instance (i.e., one model's response to one test case).
    
    **Required Columns:**
    - `task_type` (string): The type of task (e.g., `rag_faq`, `summarization`, `classification`, `chatbot`). Must match a supported task in `tasks/task_registry.py`.
    - `model` (string): An identifier for the LLM or model configuration being evaluated (e.g., `gpt-3.5-turbo`, `my_model_v2_temp0.5`).
    - `question` (string): The input prompt, question, or text provided to the LLM.
    - `ground_truth` (string): The reference answer, target summary, correct label, or ideal response.
    - `answer` (string): The actual response generated by the LLM.

    **Optional but Recommended Columns:**
    - `id` (string): A unique identifier for the test case or evaluation row. If not provided, some internal IDs might be generated.
    - `test_description` (string): A brief description of what this specific test case is evaluating.
    - `contexts` (string): For RAG tasks, this can store the context passages retrieved and provided to the LLM along with the question.
    - `ref_facts` (string): Comma-separated list of crucial facts that the `answer` should ideally contain or align with (used by `FactPresenceMetric`). Example: `"Paris is capital,Eiffel Tower is monument,Louvre is museum"`
    - `ref_key_points` (string): Comma-separated list of key topics, entities, or sub-answers the `answer` should cover (used by `ChecklistCompletenessMetric`). Example: `"Capital city,Known for,Historical significance"`
    
    **Other Custom Columns:**
    - You can include any other columns in your input data. These will be carried through to the individual scores report but generally won't be used by standard metrics unless you customize the metric logic. Example: `human_score_fluency`, `source_document_id`.
    """)
    
    st.subheader("Example Rows (Conceptual CSV/Excel Structure):")
    example_data = [
        {'id': 'rag_001', 'task_type': 'rag_faq', 'model': 'ModelAlpha', 'test_description': 'Capital of France', 'question': 'What is the capital of France?', 'contexts': 'Paris is the capital of France and a major global city...', 'ground_truth': 'Paris is the capital of France.', 'answer': 'The capital city of France is Paris.', 'ref_facts': 'Paris is capital', 'ref_key_points': 'Capital City'},
        {'id': 'sum_001', 'task_type': 'summarization', 'model': 'ModelBeta', 'test_description': 'AI Summary', 'question': 'Summarize the history of AI.', 'contexts': '', 'ground_truth': 'AI has evolved from early concepts to modern machine learning.', 'answer': 'AI started with old ideas and now uses ML.', 'ref_facts': '', 'ref_key_points': 'Evolution,Machine Learning'},
        {'id': 'cls_001', 'task_type': 'classification', 'model': 'SentimentModelX', 'test_description': 'Positive Sentiment', 'question': 'This movie was absolutely fantastic!', 'contexts': '', 'ground_truth': 'positive', 'answer': 'positive', 'ref_facts': '', 'ref_key_points': ''},
        {'id': 'chat_001', 'task_type': 'chatbot', 'model': 'HelpBot9000', 'test_description': 'Greeting', 'question': 'Hello', 'contexts': '', 'ground_truth': 'Hello! How can I assist you today?', 'answer': 'Hi there! What can I do for you?', 'ref_facts': '', 'ref_key_points': ''},
    ]
    st.dataframe(pd.DataFrame(example_data).fillna(''))
    st.markdown("""
    **Notes for CSV/Excel:**
    - Use the column headers exactly as listed (e.g., `task_type`, `ref_facts`).
    - For comma-separated fields like `ref_facts`, ensure there are no extra spaces around commas unless those spaces are part of the fact/key point itself.
    - Empty cells for optional fields are fine.
    """)


# --- Tab 4: Metrics Tutorial ---
with tab_metrics_tutorial:
    st.header("Metrics Tutorial & Explanations")
    st.markdown("Understand the metrics used in this framework. Metrics are grouped by the evaluation dimension they primarily assess. Click on a dimension to expand and see relevant metrics.")
    
    for category in CATEGORY_ORDER: # Use the defined order
        with st.expander(f"Dimension: **{category}**", expanded=(category == CAT_TRUST)): # Expand first category by default
            st.markdown(f"*{DIMENSION_DESCRIPTIONS.get(category, 'No description for this dimension.')}*")
            st.markdown("---")
            metrics_in_this_category = METRICS_BY_CATEGORY.get(category, [])
            if not metrics_in_this_category:
                st.markdown("_No metrics currently assigned to this dimension._")
            else:
                for metric_key in metrics_in_this_category:
                    info = METRIC_INFO.get(metric_key)
                    if info:
                        indicator = get_metric_indicator(metric_key)
                        st.markdown(f"##### {info['name']} (`{metric_key}`) {indicator}")
                        st.markdown(f"**Use Case & Interpretation:** {info['explanation']}")
                        
                        relevant_tasks = [task_name for task_name in get_supported_tasks() if metric_key in get_metrics_for_task(task_name)]
                        if relevant_tasks:
                            st.markdown(f"**Commonly Used For Tasks:** `{'`, `'.join(relevant_tasks)}`")
                        else:
                            st.markdown("**Commonly Used For Tasks:** (Not explicitly assigned to standard tasks in registry, may be a specialized or placeholder metric).")
                        st.markdown("---")