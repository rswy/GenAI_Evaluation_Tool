# src/file_converter.py (Flat Format - 'contexts' removed)
import pandas as pd
import json
import warnings
from pathlib import Path
import numpy as np

# --- FIELD_STRUCTURE_MAP (Updated) ---
# 'contexts' removed from input_cols for rag_faq.
# This map is primarily for conceptual understanding or if specific UI generation
# were to be based on it. The current flat file processing is more direct.
FIELD_STRUCTURE_MAP = {
    "rag_faq": {
        "input_cols": {"input_question": "question"}, # 'input_context' removed
        "reference_cols": {
            "reference_answer": "answer",
            "ref_facts": "facts",
            "ref_key_points": "key_points"
        }
    },
    "summarization": {
        "input_cols": {"input_text": "text"},
        "reference_cols": {
            "reference_summary": "summary",
            "ref_key_points": "key_points"
            }
    },
    "classification": {
        "input_cols": {"input_text": "text"},
        "reference_cols": {"reference_label": "label"}
    },
    "chatbot": {
        "input_cols": {"input_utterance": "utterance"},
        "reference_cols": {"reference_response": "response"}
    },
}


# --- Conversion Logic ---

def _process_flat_dataframe_to_data(df):
    """
    Internal helper function to process a flat pandas DataFrame
    (representing row-per-evaluation) into a list of dictionaries.
    'contexts' field will not be included if not present in df.columns.
    """
    # Define required columns for a valid evaluation row
    # Optional columns like 'id', 'ref_facts', 'ref_key_points', 'test_description' are handled later.
    # 'contexts' is no longer considered a standard column here.
    required_columns = ['task_type', 'model', 'question', 'ground_truth', 'answer']
    
    # Check for presence of required columns
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        print(f"Error: Input file must contain required columns: {required_columns}. Missing: {missing_required}")
        return None

    # Convert DataFrame to list of dictionaries, handle NaN/NaT properly
    data_list = df.to_dict('records')

    # Clean up NaN values within each dictionary record
    cleaned_data_list = []
    for record in data_list:
        cleaned_record = {}
        for key, value in record.items():
            # Replace pandas NA/NaN/NaT with None for JSON compatibility and consistency
            # If 'contexts' was a column and had NaN, it would become None. If not a column, it's skipped.
            cleaned_record[key] = None if pd.isna(value) else value
        cleaned_data_list.append(cleaned_record)

    return cleaned_data_list


# --- Public Converter Functions ---
def convert_excel_to_data(excel_path):
    """Reads Excel (flat format), converts to list of dicts."""
    try:
        excel_path = Path(excel_path)
        if not excel_path.exists(): 
            print(f"Error: Excel file not found: {excel_path}")
            return None
        # Read all as string initially to avoid type issues, pandas handles NaN correctly
        df = pd.read_excel(excel_path, dtype=str)
        # Replace pandas' default NA representation (<NA>) with None for easier processing downstream
        # Also handle numpy NaN which might come from calculations before saving
        df.replace({'<NA>': None, pd.NA: None, np.nan: None}, inplace=True)
        print(f"Read {len(df)} rows from Excel: {excel_path}")
        converted_data = _process_flat_dataframe_to_data(df)
        if converted_data is not None: 
            print(f"Converted Excel data ({len(converted_data)} cases).")
        return converted_data
    except Exception as e: 
        print(f"Excel conversion error: {e}")
        return None

def convert_csv_to_data(csv_path):
    """Reads CSV (flat format), converts to list of dicts."""
    try:
        csv_path = Path(csv_path)
        if not csv_path.exists(): 
            print(f"Error: CSV file not found: {csv_path}")
            return None
        try:
            # Read as string, keep standard NA values as NaN for pandas processing
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=True, na_values=['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'nan', 'null'])
        except UnicodeDecodeError:
             print("Warning: UTF-8 decode failed. Trying latin1 encoding for CSV.")
             df = pd.read_csv(csv_path, dtype=str, keep_default_na=True, na_values=['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'nan', 'null'], encoding='latin1')
        # Replace pandas' default NA representation (<NA>) or numpy NaN with None
        df.replace({'<NA>': None, pd.NA: None, np.nan: None}, inplace=True)
        print(f"Read {len(df)} rows from CSV: {csv_path}")
        converted_data = _process_flat_dataframe_to_data(df)
        if converted_data is not None: 
            print(f"Converted CSV data ({len(converted_data)} cases).")
        return converted_data
    except Exception as e: 
        print(f"CSV conversion error: {e}")
        return None

# Keep the main block for potential direct testing if needed
if __name__ == "__main__":
    print("Testing file converters (flat format version - 'contexts' removed)...")
    # Define project root relative to this file's location
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"

    # --- Test Excel ---
    EXCEL_TEST_FILE = data_dir / "mock_data_flat_no_contexts.xlsx" # Example filename
    try:
        # Dummy DataFrame reflecting the flat structure without 'contexts'
        dummy_data = [
            {'id': 'eval_1', 'task_type': 'rag_faq', 'model': 'model_A', 'question': 'Q1?', 'ground_truth': 'Ans1', 'answer': 'RespA1', 'ref_facts': 'fact a, fact b', 'ref_key_points': 'point x', 'test_description': 'Test RAG'},
            {'id': 'eval_2', 'task_type': 'classification', 'model': 'model_B', 'question': 'Txt1', 'ground_truth': 'positive', 'answer': 'negative', 'ref_facts': None, 'ref_key_points': None, 'test_description': 'Test Classify'},
        ]
        dummy_df_excel = pd.DataFrame(dummy_data)
        EXCEL_TEST_FILE.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        dummy_df_excel.to_excel(EXCEL_TEST_FILE, index=False)
        print(f"Created dummy Excel: {EXCEL_TEST_FILE}")
        excel_data = convert_excel_to_data(EXCEL_TEST_FILE)
        if excel_data: 
            print("First item from Excel data:")
            print(json.dumps(excel_data[0], indent=2))
    except Exception as e: 
        print(f"Could not create/process dummy excel: {e}")
        import traceback
        traceback.print_exc()


    # --- Test CSV ---
    CSV_TEST_FILE = data_dir / "mock_data_flat_no_contexts.csv"
    try:
        # Reuse dummy data structure
        dummy_df_csv = pd.DataFrame(dummy_data)
        CSV_TEST_FILE.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        dummy_df_csv.to_csv(CSV_TEST_FILE, index=False, encoding='utf-8')
        print(f"\nCreated dummy CSV: {CSV_TEST_FILE}")
        csv_data = convert_csv_to_data(CSV_TEST_FILE)
        if csv_data: 
            print("First item from CSV data:")
            print(json.dumps(csv_data[0], indent=2))
    except Exception as e: 
        print(f"Could not create/process dummy csv: {e}")
        import traceback
        traceback.print_exc()