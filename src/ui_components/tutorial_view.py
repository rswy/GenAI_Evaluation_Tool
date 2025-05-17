# src/ui_components/tutorial_view.py
import streamlit as st

# Corrected imports: Assuming 'src' is on sys.path
# Import all necessary constants from app_config
from app_config import (
    METRIC_INFO, DIMENSION_DESCRIPTIONS, CATEGORY_ORDER, 
    SEMANTIC_SIMILARITY_SCORE, METRICS_BY_CATEGORY,
    CAT_TRUST, CAT_COMPLETENESS, CAT_FLUENCY, CAT_SEMANTIC, 
    CAT_CLASSIFICATION, CAT_CONCISENESS, CAT_SAFETY, 
    CAT_PII_SAFETY, CAT_TONE, CAT_REFUSAL # Ensure all CAT_ constants are imported
)
from ui_helpers import get_metric_display_name, get_metric_indicator, is_placeholder_metric
# Import from tasks.task_registry assuming it's accessible because src is on the path
from tasks.task_registry import get_supported_tasks, get_metrics_for_task

def render_metrics_tutorial_tab():
    """Renders the 'Metrics Tutorial' tab."""
    st.header("Metrics Tutorial & Explanations")
    st.markdown("""
    Understand the metrics used in this evaluation framework. Metrics are grouped by the evaluation dimension they primarily assess.
    - Scores for most metrics range from 0 to 1. Semantic Similarity can range from -1 to 1.
    - **NaN Scores:** Indicate the metric could not be computed (e.g., missing required input data like `ref_facts`, or a calculation error).
    - **Placeholder Metrics:** These are not fully implemented and will return `NaN`. They signify areas for future development.
    - **Basic Checks:** Metrics for Safety and PII are rudimentary and not exhaustive.
    """)

    # Determine the first category to expand by default.
    default_expand_category = CATEGORY_ORDER[0] if CATEGORY_ORDER else None

    for category_tut in CATEGORY_ORDER:
        metrics_in_this_category_tut = METRICS_BY_CATEGORY.get(category_tut, [])
        if not metrics_in_this_category_tut:
            continue

        # The label of the expander is usually sufficient for Streamlit to differentiate them
        # if they are all rendered in the same scope, like this loop.
        # The `key` argument for st.expander might not be supported or needed in all versions/contexts.
        expanded_by_default = (category_tut == default_expand_category)

        # Removed the `key` argument from st.expander
        with st.expander(f"Dimension: **{category_tut}**", expanded=expanded_by_default):
            # DIMENSION_DESCRIPTIONS uses the category name (string) as key, which is `category_tut` here.
            st.markdown(f"*{DIMENSION_DESCRIPTIONS.get(category_tut, '')}*")
            st.markdown("---")

            if not metrics_in_this_category_tut:
                st.markdown("_No metrics currently assigned to this dimension._")
            else:
                for metric_key_tut in metrics_in_this_category_tut:
                    info_tut = METRIC_INFO.get(metric_key_tut)
                    if info_tut:
                        indicator_tut = get_metric_indicator(metric_key_tut) if not is_placeholder_metric(metric_key_tut) else ""
                        display_name_tut = get_metric_display_name(metric_key_tut, include_placeholder_tag=True)
                        
                        st.markdown(f"##### {display_name_tut} (`{metric_key_tut}`) {indicator_tut}")
                        
                        explanation_text_tut = info_tut['explanation']
                        if is_placeholder_metric(metric_key_tut):
                            explanation_text_tut = f"**Status: Placeholder.** {explanation_text_tut}"
                        
                        st.markdown(f"**Use Case & Interpretation:** {explanation_text_tut}")
                        
                        relevant_tasks_tut = [
                            task_name for task_name in get_supported_tasks() 
                            if metric_key_tut in get_metrics_for_task(task_name)
                        ]
                        if relevant_tasks_tut:
                            st.markdown(f"**Commonly Used For Tasks:** `{'`, `'.join(relevant_tasks_tut)}`")
                        else:
                            st.markdown("**Commonly Used For Tasks:** (General or not task-specific).")
                        
                        input_field_data_key_tut = info_tut.get("input_field_data_key")
                        if input_field_data_key_tut:
                            st.markdown(f"**Relies on Input Data Field:** `{input_field_data_key_tut}` (Score will be NaN if this field is empty/missing in a data row).")
                        
                        if metric_key_tut == SEMANTIC_SIMILARITY_SCORE: 
                             st.markdown(f"**Note:** Requires `sentence-transformers` library. May download model files on first run if not available locally and online.")
                        st.markdown("---")
