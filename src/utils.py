# src/utils.py
# General utility functions (can be expanded)

def get_nested_value(data_dict, key_path):
    """
    Retrieves a value from a nested dictionary using a dot-separated key path.

    Args:
        data_dict (dict): The dictionary to search within.
        key_path (str): A dot-separated string representing the path (e.g., 'reference.answer').

    Returns:
        object: The value found at the specified path, or None if the path is invalid or missing.
    """
    if not isinstance(data_dict, dict) or not isinstance(key_path, str):
        return None

    keys = key_path.split('.')
    value = data_dict
    try:
        for key in keys:
            if isinstance(value, dict):
                 value = value.get(key) # Use .get for safer access, returns None if key missing
                 if value is None:
                      return None # Stop if any key is missing along the path
            else:
                 return None # Cannot traverse further if not a dict
        return value
    except Exception: # Catch any unexpected errors during access
        return None

