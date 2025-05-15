# src/data_loader.py
import json
import pandas as pd # Keep pandas in case needed elsewhere, though not used here now
from pathlib import Path

def load_data(file_path):
    """
    Loads evaluation data from a JSON file.

    Args:
        file_path (str or Path): Path to the JSON data file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a test case.
              Returns None if the file is not found or is invalid.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        print(f"Error: Data file not found at {file_path}")
        return None # Return None on error
    try:
        with open(file_path, 'r', encoding='utf-8') as f: # Specify encoding
            data = json.load(f)
        if not isinstance(data, list):
            print(f"Error: Data file {file_path} should contain a JSON list.")
            return None # Return None on error
        # Basic validation: check if items are dictionaries
        if data and not all(isinstance(item, dict) for item in data):
             print(f"Error: Items in the JSON list in {file_path} should be dictionaries.")
             return None
        return data
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {file_path}: {e}")
        return None # Return None on error
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        return None # Return None on error
