# src/file_converter.py (Flat Format)
import pandas as pd
import json
import warnings
from pathlib import Path
import numpy as np

# --- Keep FIELD_STRUCTURE_MAP for Streamlit form reference ---
# Although the conversion logic is simpler now, this map is still
# useful for defining the expected structure for different tasks,
# especially for the "Add Test Case" form in Streamlit.
FIELD_STRUCTURE_MAP = {
    "rag_faq": {
        "input_cols": {"input_question": "question", "input_context": "context"},
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
    """
    # Define required columns for a valid evaluation row
    # Optional columns like 'id', 'contexts', 'ref_facts', 'ref_key_points' are handled later
    required_columns = ['task_type', 'model', 'question', 'ground_truth', 'answer']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        print(f"Error: Input file must contain required columns: {required_columns}. Missing: {missing}")
        # Optionally check for task-specific input columns based on FIELD_STRUCTURE_MAP if needed
        # For example, check if 'input_context' exists for 'rag_faq' tasks
        return None

    # Convert DataFrame to list of dictionaries, handle NaN/NaT properly
    # Convert to dictionary records first
    data_list = df.to_dict('records')

    # Clean up NaN values within each dictionary record
    cleaned_data_list = []
    for record in data_list:
        cleaned_record = {}
        for key, value in record.items():
            # Replace pandas NA/NaN/NaT with None for JSON compatibility and consistency
            cleaned_record[key] = None if pd.isna(value) else value
        cleaned_data_list.append(cleaned_record)

    return cleaned_data_list


# --- Public Converter Functions ---
def convert_excel_to_data(excel_path):
    """Reads Excel (flat format), converts to list of dicts."""
    try:
        excel_path = Path(excel_path)
        if not excel_path.exists(): print(f"Error: Excel file not found: {excel_path}"); return None
        # Read all as string initially to avoid type issues, pandas handles NaN correctly
        df = pd.read_excel(excel_path, dtype=str)
        # Replace pandas' default NA representation (<NA>) with None for easier processing downstream
        # Also handle numpy NaN which might come from calculations before saving
        df.replace({'<NA>': None, pd.NA: None, np.nan: None}, inplace=True)
        print(f"Read {len(df)} rows from Excel: {excel_path}")
        converted_data = _process_flat_dataframe_to_data(df)
        if converted_data is not None: print(f"Converted Excel data ({len(converted_data)} cases).")
        return converted_data
    except Exception as e: print(f"Excel conversion error: {e}"); return None

def convert_csv_to_data(csv_path):
    """Reads CSV (flat format), converts to list of dicts."""
    try:
        csv_path = Path(csv_path)
        if not csv_path.exists(): print(f"Error: CSV file not found: {csv_path}"); return None
        try:
            # Read as string, keep standard NA values as NaN for pandas processing
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=True, na_values=['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'nan', 'null'])
        except UnicodeDecodeError:
             print("Warning: UTF-8 decode failed. Trying latin1 encoding for CSV.");
             df = pd.read_csv(csv_path, dtype=str, keep_default_na=True, na_values=['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'nan', 'null'], encoding='latin1')
        # Replace pandas' default NA representation (<NA>) or numpy NaN with None
        df.replace({'<NA>': None, pd.NA: None, np.nan: None}, inplace=True)
        print(f"Read {len(df)} rows from CSV: {csv_path}")
        converted_data = _process_flat_dataframe_to_data(df)
        if converted_data is not None: print(f"Converted CSV data ({len(converted_data)} cases).")
        return converted_data
    except Exception as e: print(f"CSV conversion error: {e}"); return None

# Keep the main block for potential direct testing if needed
if __name__ == "__main__":
    print("Testing file converters (flat format version)...")
    # Define project root relative to this file's location
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"

    # --- Test Excel ---
    EXCEL_TEST_FILE = data_dir / "mock_data_flat.xlsx" # Example filename
    try:
        # Create a dummy DataFrame reflecting the new flat structure
        dummy_data = [
            {'id': 'eval_1', 'task_type': 'rag_faq', 'model': 'model_A', 'question': 'Q1?', 'contexts': 'Ctx1', 'ground_truth': 'Ans1', 'answer': 'RespA1', 'ref_facts': 'fact a, fact b', 'ref_key_points': 'point x'},
            {'id': 'eval_2', 'task_type': 'classification', 'model': 'model_B', 'question': 'Txt1', 'contexts': None, 'ground_truth': 'positive', 'answer': 'negative', 'ref_facts': None, 'ref_key_points': None},
        ]
        dummy_df_excel = pd.DataFrame(dummy_data)
        EXCEL_TEST_FILE.parent.mkdir(exist_ok=True)
        dummy_df_excel.to_excel(EXCEL_TEST_FILE, index=False)
        print(f"Created dummy Excel: {EXCEL_TEST_FILE}")
        excel_data = convert_excel_to_data(EXCEL_TEST_FILE)
        if excel_data: print(json.dumps(excel_data[:1], indent=2))
    except Exception as e: print(f"Could not create/process dummy excel: {e}")

    # --- Test CSV ---
    CSV_TEST_FILE = data_dir / "mock_data_flat.csv"
    try:
        # Reuse dummy data structure
        dummy_df_csv = pd.DataFrame(dummy_data)
        CSV_TEST_FILE.parent.mkdir(exist_ok=True)
        dummy_df_csv.to_csv(CSV_TEST_FILE, index=False, encoding='utf-8')
        print(f"\nCreated dummy CSV: {CSV_TEST_FILE}")
        csv_data = convert_csv_to_data(CSV_TEST_FILE)
        if csv_data: print(json.dumps(csv_data[:1], indent=2))
    except Exception as e: print(f"Could not create/process dummy csv: {e}")


# # src/file_converter.py (Reverted)
# import pandas as pd
# import json # Keep for potential future use
# import warnings
# from pathlib import Path

# # --- Configuration: Define how Excel/CSV columns map to JSON structure ---
# # *** Reverted: Removed 'ref_facts' and 'ref_key_points' mapping ***
# FIELD_STRUCTURE_MAP = {
#     "rag_faq": {
#         "input_cols": {"input_question": "question", "input_context": "context"},
#         "reference_cols": {"reference_answer": "answer"} # Only primary answer
#     },
#     "summarization": {
#         "input_cols": {"input_text": "text"},
#         "reference_cols": {"reference_summary": "summary"} # Only primary summary
#     },
#     "classification": {
#         "input_cols": {"input_text": "text"},
#         "reference_cols": {"reference_label": "label"}
#     },
#     "chatbot": {
#         "input_cols": {"input_utterance": "utterance"},
#         "reference_cols": {"reference_response": "response"}
#     },
#     # Add mappings for any new task types here
# }

# # --- Shared Conversion Logic ---
# def _process_dataframe_to_data(df):
#     """
#     Internal helper function to process a pandas DataFrame into the
#     list-of-dictionaries test case format. (Reverted reference handling)
#     """
#     output_data = []
#     processed_ids = set()
#     if 'id' not in df.columns or 'task_type' not in df.columns:
#         print("Error: Input file must contain 'id' and 'task_type' columns.")
#         return None

#     for index, row in df.iterrows():
#         test_case = {}
#         task_type_val = row.get('task_type'); case_id_val = row.get('id')
#         if pd.isna(task_type_val) or pd.isna(case_id_val):
#              warnings.warn(f"Skipping row {index + 2}: Missing 'id' or 'task_type'.", RuntimeWarning); continue
#         task_type = str(task_type_val); case_id = str(case_id_val)
#         if case_id in processed_ids:
#              warnings.warn(f"Duplicate ID '{case_id}' in row {index + 2}. Skipping.", RuntimeWarning); continue
#         processed_ids.add(case_id)

#         test_case['id'] = case_id; test_case['task_type'] = task_type
#         task_map = FIELD_STRUCTURE_MAP.get(task_type)
#         test_case['input'] = {}; test_case['reference'] = {}; test_case['llm_responses'] = {} # Initialize

#         if task_map:
#             # Map Input Fields
#             for col_name, json_key in task_map.get("input_cols", {}).items():
#                 if col_name in df.columns: test_case['input'][json_key] = "" if pd.isna(row[col_name]) else str(row[col_name])
#             # Map Reference Fields (Reverted: only maps columns defined in reference_cols)
#             for col_name, json_key in task_map.get("reference_cols", {}).items():
#                 if col_name in df.columns: test_case['reference'][json_key] = "" if pd.isna(row[col_name]) else str(row[col_name])
#         else: warnings.warn(f"Task type '{task_type}' in row {index + 2} not in FIELD_STRUCTURE_MAP.", RuntimeWarning)

#         # Map LLM Responses
#         for col in df.columns:
#             if col.startswith("response_"):
#                 model_config_name = col[len("response_"):]
#                 if model_config_name: test_case['llm_responses'][model_config_name] = "" if pd.isna(row[col]) else str(row[col])
#                 else: warnings.warn(f"Column '{col}' invalid response format.", RuntimeWarning)

#         # Map Human Scores (or other pass-through columns) - Kept for flexibility
#         for col in df.columns:
#             if col.startswith("human_score_"): test_case[col] = "" if pd.isna(row[col]) else str(row[col])

#         output_data.append(test_case)
#     return output_data

# # --- Public Converter Functions ---
# def convert_excel_to_data(excel_path):
#     """Reads Excel, converts to test case data structure."""
#     try:
#         excel_path = Path(excel_path)
#         if not excel_path.exists(): print(f"Error: Excel file not found: {excel_path}"); return None
#         df = pd.read_excel(excel_path)
#         print(f"Read {len(df)} rows from Excel: {excel_path}")
#         converted_data = _process_dataframe_to_data(df)
#         if converted_data is not None: print(f"Converted Excel data ({len(converted_data)} cases).")
#         return converted_data
#     except Exception as e: print(f"Excel conversion error: {e}"); return None

# def convert_csv_to_data(csv_path):
#     """Reads CSV, converts to test case data structure."""
#     try:
#         csv_path = Path(csv_path)
#         if not csv_path.exists(): print(f"Error: CSV file not found: {csv_path}"); return None
#         try:
#             df = pd.read_csv(csv_path, dtype=str, keep_default_na=True, na_values=['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'nan', 'null'])
#         except UnicodeDecodeError:
#              print("Warning: UTF-8 decode failed. Trying latin1 encoding for CSV.");
#              df = pd.read_csv(csv_path, dtype=str, keep_default_na=True, na_values=['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'nan', 'null'], encoding='latin1')
#         print(f"Read {len(df)} rows from CSV: {csv_path}")
#         converted_data = _process_dataframe_to_data(df)
#         if converted_data is not None: print(f"Converted CSV data ({len(converted_data)} cases).")
#         return converted_data
#     except Exception as e: print(f"CSV conversion error: {e}"); return None

# # Keep the main block for potential direct testing if needed
# if __name__ == "__main__":
#     # Example Usage: Convert files and print results (using reverted structure)
#     print("Testing file converters (reverted version)...")
#     # (Testing logic can remain similar, but generated files won't have facts/key_points)









# # src/file_converter.py
# import pandas as pd
# import json # Keep for potential future use
# import warnings
# from pathlib import Path

# # --- Configuration: Define how Excel/CSV columns map to JSON structure ---
# # *** Added example column names 'ref_facts' and 'ref_key_points' ***
# # Users need to name their columns exactly like this in Excel/CSV.
# FIELD_STRUCTURE_MAP = {
#     "rag_faq": {
#         "input_cols": {"input_question": "question", "input_context": "context"},
#         # Map Excel/CSV column names to the keys expected by the evaluator/metrics
#         "reference_cols": {
#             "reference_answer": "answer",
#             "ref_facts": "facts", # Maps 'ref_facts' column to 'facts' key in reference dict
#             "ref_key_points": "key_points" # Maps 'ref_key_points' column to 'key_points' key
#         }
#     },
#     "summarization": {
#         "input_cols": {"input_text": "text"},
#         "reference_cols": {
#             "reference_summary": "summary",
#             "ref_key_points": "key_points" # Added key points mapping
#             }
#     },
#     "classification": {
#         "input_cols": {"input_text": "text"},
#         "reference_cols": {"reference_label": "label"}
#     },
#     "chatbot": {
#         "input_cols": {"input_utterance": "utterance"},
#         "reference_cols": {"reference_response": "response"}
#         # Could add mappings for facts/key_points here if needed for chatbot task
#         # "ref_facts": "facts",
#         # "ref_key_points": "key_points"
#     },
#     # Add mappings for any new task types here
# }

# # --- Shared Conversion Logic ---
# # (No changes needed in _process_dataframe_to_data, it uses the map above)
# def _process_dataframe_to_data(df):
#     """
#     Internal helper function to process a pandas DataFrame into the
#     list-of-dictionaries test case format.
#     """
#     output_data = []
#     processed_ids = set()

#     # Check for required columns after loading
#     if 'id' not in df.columns or 'task_type' not in df.columns:
#         print("Error: Input file must contain 'id' and 'task_type' columns.")
#         return None

#     for index, row in df.iterrows():
#         test_case = {}
#         # Read task_type and id first
#         task_type_val = row.get('task_type')
#         case_id_val = row.get('id')

#         # Basic validation
#         if pd.isna(task_type_val) or pd.isna(case_id_val):
#              warnings.warn(f"Skipping row {index + 2}: Missing required 'id' or 'task_type'.", RuntimeWarning)
#              continue

#         task_type = str(task_type_val)
#         case_id = str(case_id_val) # Ensure ID is string

#         # Prevent duplicate IDs
#         if case_id in processed_ids:
#              warnings.warn(f"Duplicate ID '{case_id}' found in row {index + 2}. Skipping this row.", RuntimeWarning)
#              continue
#         processed_ids.add(case_id)

#         test_case['id'] = case_id
#         test_case['task_type'] = task_type

#         task_map = FIELD_STRUCTURE_MAP.get(task_type) # Use .get for safer access

#         # Initialize keys even if map is missing
#         test_case['input'] = {}
#         test_case['reference'] = {}
#         test_case['llm_responses'] = {}

#         if task_map:
#              # --- Map Input Fields ---
#             for col_name, json_key in task_map.get("input_cols", {}).items():
#                 if col_name in df.columns:
#                     test_case['input'][json_key] = "" if pd.isna(row[col_name]) else str(row[col_name])
#                 else:
#                     # Optional: Warn if expected input column is missing
#                     # warnings.warn(f"Input column '{col_name}' defined for task '{task_type}' not found. Skipping.", RuntimeWarning)
#                     pass # Silently skip if column not present

#             # --- Map Reference Fields ---
#             for col_name, json_key in task_map.get("reference_cols", {}).items():
#                 if col_name in df.columns:
#                     # Store the value as read (could be string list, comma-sep string, etc.)
#                     # The metric will handle parsing later
#                     test_case['reference'][json_key] = "" if pd.isna(row[col_name]) else str(row[col_name])
#                 else:
#                     # Optional: Warn if expected reference column is missing
#                     # warnings.warn(f"Reference column '{col_name}' defined for task '{task_type}' not found. Skipping.", RuntimeWarning)
#                     pass # Silently skip if column not present
#         else:
#              warnings.warn(f"Task type '{task_type}' in row {index + 2} not found in FIELD_STRUCTURE_MAP. Input/Reference mapping may be incomplete.", RuntimeWarning)


#         # --- Map LLM Responses (Applies regardless of task map) ---
#         for col in df.columns:
#             if col.startswith("response_"):
#                 model_config_name = col[len("response_"):]
#                 if model_config_name:
#                     test_case['llm_responses'][model_config_name] = "" if pd.isna(row[col]) else str(row[col]) # Ensure string
#                 else:
#                     warnings.warn(f"Column '{col}' starts with 'response_' but has no name after it. Skipping.", RuntimeWarning)

#         # --- Map Human Scores (or other pass-through columns) ---
#         # Add any other columns directly to the top level or a specific key if needed
#         # Example: Add columns starting with 'human_score_'
#         for col in df.columns:
#             if col.startswith("human_score_"):
#                  # Store human scores at the top level of the test case for simplicity
#                  test_case[col] = "" if pd.isna(row[col]) else str(row[col])


#         output_data.append(test_case)

#     return output_data


# # --- Public Converter Functions ---
# # (No changes needed in convert_excel_to_data or convert_csv_to_data bodies)
# def convert_excel_to_data(excel_path):
#     """
#     Reads an Excel file (.xlsx) and converts it to the test case data structure.
#     """
#     try:
#         excel_path = Path(excel_path)
#         if not excel_path.exists():
#             print(f"Error: Excel file not found at {excel_path}")
#             return None

#         df = pd.read_excel(excel_path)
#         print(f"Read {len(df)} rows from Excel file: {excel_path}")
#         converted_data = _process_dataframe_to_data(df)
#         if converted_data is not None:
#              print(f"Successfully converted Excel data ({len(converted_data)} cases).")
#         return converted_data

#     except FileNotFoundError:
#          print(f"Error: Input Excel file not found at {excel_path}")
#          return None
#     except Exception as e:
#         print(f"An unexpected error occurred during Excel conversion: {e}")
#         return None


# def convert_csv_to_data(csv_path):
#     """
#     Reads a CSV file (.csv) and converts it to the test case data structure.
#     """
#     try:
#         csv_path = Path(csv_path)
#         if not csv_path.exists():
#             print(f"Error: CSV file not found at {csv_path}")
#             return None

#         try:
#             # Keep na_values handling as before, ensure dtype=str
#             df = pd.read_csv(csv_path, dtype=str, keep_default_na=True, na_values=['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'nan', 'null'])
#         except UnicodeDecodeError:
#              print("Warning: UTF-8 decoding failed. Trying latin1 encoding for CSV.")
#              df = pd.read_csv(csv_path, dtype=str, keep_default_na=True, na_values=['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'nan', 'null'], encoding='latin1')


#         print(f"Read {len(df)} rows from CSV file: {csv_path}")
#         converted_data = _process_dataframe_to_data(df)
#         if converted_data is not None:
#             print(f"Successfully converted CSV data ({len(converted_data)} cases).")
#         return converted_data

#     except FileNotFoundError:
#          print(f"Error: Input CSV file not found at {csv_path}")
#          return None
#     except Exception as e:
#         print(f"An unexpected error occurred during CSV conversion: {e}")
#         return None