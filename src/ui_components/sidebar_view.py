# src/ui_components/sidebar_view.py
import streamlit as st
from pathlib import Path
import os
import tempfile
import warnings
import pandas as pd
# Assuming these are moved to app_config or passed if needed
# from ..app_config import ...
# For this example, direct imports from framework if needed by data loading
from ..data_loader import load_data
from ..file_converter import convert_excel_to_data, convert_csv_to_data
from ..mock_data_generator import generate_mock_data_flat
from ..ui_helpers import clear_app_state # Assuming clear_app_state is in ui_helpers

def render_sidebar():
    """Renders the sidebar for data input options."""
    st.sidebar.header("⚙️ Input Options")

    # This callback is defined here as it directly uses st.session_state
    # and clear_app_state which might also manipulate session_state.
    def on_input_method_change():
        clear_app_state() # Clear previous data when switching input methods

    input_method = st.sidebar.radio(
        "Choose data source:", ("Upload File", "Generate Mock Data"),
        key="input_method_radio",
        on_change=on_input_method_change
    )

    if input_method == "Upload File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload (.xlsx, .csv, .json - Flat Format)", type=["xlsx", "csv", "json"],
            key="file_uploader_widget"
        )
        if uploaded_file is not None:
            # If a new file is uploaded, process it.
            # The clear_app_state on radio change handles clearing old data.
            # We only re-process if the filename changes or if no data is loaded yet from this file.
            if uploaded_file.name != st.session_state.get('last_uploaded_file_name') or st.session_state.get('edited_test_cases_df').empty:
                st.session_state.last_uploaded_file_name = uploaded_file.name
                file_suffix = Path(uploaded_file.name).suffix.lower()
                st.session_state.data_source_info = f"Processing: {uploaded_file.name}"
                st.sidebar.info(st.session_state.data_source_info) # Immediate feedback
                
                tmp_file_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = Path(tmp_file.name)
                    
                    test_data_list_from_file = None
                    with st.spinner(f"Loading and converting {uploaded_file.name}..."):
                        if file_suffix == ".xlsx": test_data_list_from_file = convert_excel_to_data(tmp_file_path)
                        elif file_suffix == ".csv": test_data_list_from_file = convert_csv_to_data(tmp_file_path)
                        elif file_suffix == ".json": test_data_list_from_file = load_data(tmp_file_path)
                    
                    if test_data_list_from_file:
                        st.session_state.test_cases_list_loaded = test_data_list_from_file
                        df_for_edit = pd.DataFrame(test_data_list_from_file)
                        required_cols_editor = ['id', 'task_type', 'model', 'question', 'ground_truth', 'answer', 
                                                'ref_facts', 'ref_key_points', 'test_description', 'contexts'] 
                        for col in required_cols_editor:
                            if col not in df_for_edit.columns: df_for_edit[col] = None 
                        st.session_state.edited_test_cases_df = df_for_edit.copy().fillna('')
                        st.session_state.data_source_info = f"Loaded {len(test_data_list_from_file)} rows from {uploaded_file.name} into editor."
                        # No st.sidebar.success here, let main app handle status display
                    else:
                        if test_data_list_from_file == []: raise ValueError("File loaded but was empty or contained no valid data rows.")
                        else: raise ValueError("Failed to load/convert data. Check file format, content, and required columns.")
                except Exception as e:
                    st.session_state.data_source_info = f"Error processing {uploaded_file.name}: {e}"
                    # Error will be displayed in the main app area
                    st.session_state.edited_test_cases_df = pd.DataFrame() # Clear editor on error
                finally:
                    if tmp_file_path and tmp_file_path.exists():
                        try: os.unlink(tmp_file_path)
                        except Exception as e_unlink: warnings.warn(f"Could not delete temp file {tmp_file_path}: {e_unlink}")
            # If same file is re-uploaded or already processed, data_source_info might still be set
            # The main app will display this st.session_state.data_source_info

    elif input_method == "Generate Mock Data":
        st.sidebar.warning("Mock data provides example flat-format rows with varied answer quality.")
        if st.sidebar.button("Generate and Use Mock Data", key="generate_mock_button"):
            # clear_app_state() # Already called by on_change of radio button
            try:
                with st.spinner("Generating mock evaluation data..."):
                    mock_data_list = generate_mock_data_flat(num_samples_per_task=3) 
                if mock_data_list:
                    st.session_state.test_cases_list_loaded = mock_data_list
                    df_for_edit = pd.DataFrame(mock_data_list)
                    required_cols_editor = ['id', 'task_type', 'model', 'question', 'ground_truth', 'answer', 
                                            'ref_facts', 'ref_key_points', 'test_description', 'contexts']
                    for col in required_cols_editor:
                        if col not in df_for_edit.columns: df_for_edit[col] = None
                    st.session_state.edited_test_cases_df = df_for_edit.copy().fillna('')
                    st.session_state.data_source_info = f"Using {len(mock_data_list)} generated mock rows, loaded into editor."
                else: 
                    st.session_state.data_source_info = "Failed to generate mock data."
            except Exception as e:
                st.session_state.data_source_info = f"Error generating mock data: {e}"
                # traceback will be shown in main app if error occurs during button press

    # Display data source info in the main app area, not sidebar
    if st.session_state.data_source_info:
        if "error" in st.session_state.data_source_info.lower() or "failed" in st.session_state.data_source_info.lower():
            st.error(st.session_state.data_source_info)
            # If a traceback was stored, show it
            if "traceback" in st.session_state:
                 st.text_area("Traceback", st.session_state.traceback, height=150)
                 del st.session_state.traceback # Clear after showing
        elif "loaded" in st.session_state.data_source_info.lower() or "using" in st.session_state.data_source_info.lower():
            st.success(st.session_state.data_source_info)
        else:
            st.info(st.session_state.data_source_info)

