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
st.info("ℹ️ How to use testing framework: \n\n" \
"- Adding Test Cases: " \
"\n\n\t - Using the UI, go to View/Edit/Add Data Tab to key in your test cases. " \
"\n\n\t - Using an existing csv/json/xlsx file with the required columns, click on upload File on the sidebar and browser to upload file." \
"\n\n\t - Using mock data, click on 'Generate Mock Data' on the sidebar and click 'Generate and use Mock Data. \n\n" \
"- Running Evaluation on Data: " \
"\n\n\t - Once the data is uploaded/added, under the 'Evaluation & Results' tab, click 'Run Evaluation on Data in Editor' button. \n\n" \
"\n- Refer to 'README.md' for even more details. \n\n" \
"-------------------------------\n\n" \
"ℹ️ How to use Command Line instead to run batch test cases: Refer to 'how_to_use_mainpy.md' \n\n \n\n" \
"-------------------------------\n\n" \

"ℹ️ Quick Guides to Data Formatting and Metrics, Refer to the right two tabs below ⬇️ \n\n")


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
    st.info("ℹ️ **Semantic Similarity Note:** Running Semantic Similarity can take slightly more computation time. Please be patient. Also, If this is your first time running with Semantic Similarity, "
        "the 'sentence-transformers' library may need to download model files (e.g., 'all-MiniLM-L6-v2'), "
        "which requires an internet connection. This download happens once per model. "
        "Subsequent runs will use the cached model. For offline use, pre-download the model (see documentation/tool output)")

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

