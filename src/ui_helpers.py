# src/ui_helpers.py
"""
Utility functions for the Streamlit UI, such as formatting metric names,
applying styles, and managing session state.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.cm as cm
import warnings
import copy

# Import METRIC_INFO from app_config
# Assuming app_config.py is in the same directory (src)
from app_config import METRIC_INFO, SEMANTIC_SIMILARITY_SCORE


def get_metric_display_name(metric_key, include_placeholder_tag=True):
    """Gets the display-friendly name for a metric."""
    info = METRIC_INFO.get(metric_key, {})
    name = info.get('name', metric_key.replace('_', ' ').title())
    if include_placeholder_tag and info.get("status") == "placeholder":
        if "(Placeholder)" not in name:  # Avoid double tagging
            name += " (Placeholder)"
    return name

def get_metric_indicator(metric_key):
    """Gets the up/down arrow indicator for a metric based on whether higher is better."""
    info = METRIC_INFO.get(metric_key)
    if not info:
        return ""
    return "⬆️" if info.get("higher_is_better", True) else "⬇️"


def is_placeholder_metric(metric_key):
    """Checks if a metric is a placeholder."""
    info = METRIC_INFO.get(metric_key, {})
    return info.get("status") == "placeholder"

def apply_color_gradient(styler, metric_info_config=METRIC_INFO):
    """
    Applies color gradient styling to a Pandas Styler object based on metric properties.
    Args:
        styler: Pandas Styler object.
        metric_info_config: The METRIC_INFO dictionary.
    """
    cmap_good = matplotlib.colormaps.get_cmap('RdYlGn')
    cmap_bad = matplotlib.colormaps.get_cmap('RdYlGn_r')

    for col_name_display in styler.columns:
        if not isinstance(col_name_display, str):
            continue
        original_metric_key = None
        for mk_orig, m_info_orig in metric_info_config.items():
            possible_names = [
                f"{get_metric_display_name(mk_orig, True)} {get_metric_indicator(mk_orig)}".strip(),
                f"{get_metric_display_name(mk_orig, False)} {get_metric_indicator(mk_orig)}".strip(),
                get_metric_display_name(mk_orig, True).strip(),
                get_metric_display_name(mk_orig, False).strip(),
                mk_orig
            ]
            if col_name_display.strip() in possible_names:
                original_metric_key = mk_orig
                break
        
        info = metric_info_config.get(original_metric_key)

        if info and pd.api.types.is_numeric_dtype(styler.data[col_name_display]) and not is_placeholder_metric(original_metric_key):
            cmap_to_use = cmap_good if info['higher_is_better'] else cmap_bad
            try:
                data_col = styler.data[col_name_display].dropna().astype(float)
                if data_col.empty:
                    continue
                vmin = data_col.min()
                vmax = data_col.max()

                gradient_vmin = 0.0
                gradient_vmax = 1.0
                if original_metric_key == SEMANTIC_SIMILARITY_SCORE:
                    gradient_vmin = min(data_col.min(), -1.0) 
                    gradient_vmax = max(data_col.max(), 1.0)

                if np.isclose(vmin, vmax):
                    mid_point_norm = (gradient_vmin + gradient_vmax) / 2.0
                    if (gradient_vmax - gradient_vmin) == 0:
                        norm_val = 0.5 
                    else:
                        norm_val = (vmin - gradient_vmin) / (gradient_vmax - gradient_vmin)
                    
                    color_val_single = norm_val
                    if not info['higher_is_better']:
                        color_val_single = 1.0 - norm_val
                    
                    styler.background_gradient(cmap=matplotlib.colors.ListedColormap([cmap_to_use(color_val_single)]), subset=[col_name_display])
                else:
                    styler.background_gradient(cmap=cmap_to_use, subset=[col_name_display], vmin=gradient_vmin, vmax=gradient_vmax)
            except Exception as e:
                warnings.warn(f"Style error for '{col_name_display}' (orig key: {original_metric_key}): {e}", RuntimeWarning)

    format_dict = {}
    for col_disp_name_format in styler.columns:
        if isinstance(col_disp_name_format, str):
            original_mkey_for_format = None
            for mk_orig_fmt, m_info_orig_fmt in metric_info_config.items():
                possible_names_fmt = [
                    f"{get_metric_display_name(mk_orig_fmt, True)} {get_metric_indicator(mk_orig_fmt)}".strip(),
                    f"{get_metric_display_name(mk_orig_fmt, False)} {get_metric_indicator(mk_orig_fmt)}".strip(),
                    get_metric_display_name(mk_orig_fmt, True).strip(),
                    get_metric_display_name(mk_orig_fmt, False).strip(),
                    mk_orig_fmt
                ]
                if col_disp_name_format.strip() in possible_names_fmt:
                    original_mkey_for_format = mk_orig_fmt
                    break
            
            if original_mkey_for_format and pd.api.types.is_numeric_dtype(styler.data[col_disp_name_format]):
                 format_dict[col_disp_name_format] = '{:.4f}'
            elif col_disp_name_format in ['Observations', 'Potential Actions', 'Metrics Not Computed or Not Applicable']:
                pass
            elif styler.data[col_disp_name_format].dtype == 'object' and any(isinstance(x, float) for x in styler.data[col_disp_name_format].dropna()):
                 format_dict[col_disp_name_format] = lambda x: f"{x:.4f}" if isinstance(x, float) else x

    styler.format(formatter=format_dict, na_rep='NaN')
    return styler


def unflatten_df_to_test_cases(df):
    """Converts a flattened DataFrame back to a list of test case dictionaries."""
    test_cases_list = []
    if df is None or df.empty:
        return []
    
    direct_keys = ['id', 'task_type', 'model', 'question', 'ground_truth', 'answer',
                   'ref_facts', 'ref_key_points', 'test_description', 'contexts']

    for _, row_series in df.iterrows():
        row = row_series.to_dict()
        case = {}
        # Check for essential fields
        if pd.isna(row.get('task_type')) or pd.isna(row.get('model')) or \
           pd.isna(row.get('question')) or pd.isna(row.get('ground_truth')) or \
           pd.isna(row.get('answer')):
            warnings.warn(f"Skipping row due to missing required field(s) (task_type, model, question, ground_truth, answer): {row.get('id', 'Unknown ID')}")
            continue
        
        for key in direct_keys:
            if key in row and pd.notna(row[key]):
                case[key] = str(row[key])
            elif key in row:  # Key exists but is NaN/None
                case[key] = None
        
        for col_name, value in row.items():
            if col_name not in case:  # Avoid overwriting already processed keys
                case[col_name] = str(value) if pd.notna(value) else None
        
        test_cases_list.append(case)
    return test_cases_list

# --- Session State Management ---
def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    default_state_keys = {
        'test_cases_list_loaded': None, 'edited_test_cases_df': pd.DataFrame(),
        'aggregated_results_df': None, 'individual_scores_df': None,
        'data_source_info': None, 'last_uploaded_file_name': None,
        'metrics_for_agg_display': [],
        'add_row_input_mode': "Easy (Required Fields Only)"
    }
    for key, default_value in default_state_keys.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(default_value)

def clear_app_state():
    """Clears the main application state variables in session_state."""
    st.session_state.test_cases_list_loaded = None
    st.session_state.edited_test_cases_df = pd.DataFrame()
    st.session_state.aggregated_results_df = None
    st.session_state.individual_scores_df = None
    st.session_state.data_source_info = None
    st.session_state.metrics_for_agg_display = []
    st.session_state.last_uploaded_file_name = None
    # st.session_state.add_row_input_mode = "Easy (Required Fields Only)" # Optionally reset this too
