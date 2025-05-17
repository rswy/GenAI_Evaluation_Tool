# src/ui_components/results_view.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import defaultdict
import datetime

# Corrected imports: Assuming 'src' is on sys.path
from app_config import METRIC_INFO, KEY_METRICS_PER_TASK_FOR_HIGHLIGHTS, CATEGORY_ORDER
from ui_helpers import get_metric_display_name, get_metric_indicator, is_placeholder_metric, apply_color_gradient
from tasks.task_registry import get_metrics_for_task

def render_individual_scores_sub_tab():
    """Renders the 'Individual Scores' sub-tab within 'Evaluation & Results'."""
    st.subheader("📊 Individual Test Case Scores")
    if st.session_state.individual_scores_df is not None and not st.session_state.individual_scores_df.empty:
        ind_df_display = st.session_state.individual_scores_df.copy()
        renamed_cols_ind_display = {}
        original_metric_keys_in_ind_df = [col for col in ind_df_display.columns if col in METRIC_INFO]
        for m_key in original_metric_keys_in_ind_df:
            renamed_cols_ind_display[m_key] = f"{get_metric_display_name(m_key, True)} {get_metric_indicator(m_key) if not is_placeholder_metric(m_key) else ''}".strip()
        ind_df_display.rename(columns=renamed_cols_ind_display, inplace=True)

        id_cols = ['id', 'task_type', 'model', 'test_description']
        metric_cols_display_names = [renamed_cols_ind_display.get(m_key, m_key) for m_key in original_metric_keys_in_ind_df if renamed_cols_ind_display.get(m_key, m_key) in ind_df_display.columns]
        interpretation_cols = ['Observations', 'Potential Actions', 'Metrics Not Computed or Not Applicable']
        input_output_cols = ['question', 'ground_truth', 'answer', 'ref_facts', 'ref_key_points', 'contexts']
        
        final_order_ind_display = []
        for col_group in [id_cols, metric_cols_display_names, interpretation_cols, input_output_cols]:
            for col in col_group:
                if col in ind_df_display.columns and col not in final_order_ind_display: 
                    final_order_ind_display.append(col)
        remaining_other_cols_ind = sorted([col for col in ind_df_display.columns if col not in final_order_ind_display and not col.startswith('_st_')])
        final_order_ind_display.extend(remaining_other_cols_ind)
        final_order_ind_display = [col for col in final_order_ind_display if col in ind_df_display.columns] 

        st.info("Displaying all scores and interpretations for each test case. Use column headers to sort. Download full table below.")
        st.dataframe(ind_df_display[final_order_ind_display].style.pipe(apply_color_gradient, METRIC_INFO), use_container_width=True)
        
        st.divider()
        st.subheader("🔍 Detailed Interpretation for a Single Test Case")
        if 'id' in st.session_state.individual_scores_df.columns:
            available_ids = st.session_state.individual_scores_df['id'].astype(str).unique().tolist()
            if not available_ids: st.warning("No test case IDs found.")
            else:
                options_for_selectbox = ["<Select a Test Case ID>"] + available_ids
                selected_id_for_interp = st.selectbox(
                    "Select Test Case ID:", options=options_for_selectbox, index=0, key="individual_case_interp_selector_results_tab"
                )
                if selected_id_for_interp and selected_id_for_interp != "<Select a Test Case ID>":
                    selected_case_data = st.session_state.individual_scores_df[st.session_state.individual_scores_df['id'].astype(str) == selected_id_for_interp]
                    if not selected_case_data.empty:
                        case_to_show = selected_case_data.iloc[0] 
                        st.markdown(f"**Test Case ID:** `{case_to_show.get('id', 'N/A')}`")
                        st.markdown(f"**Model:** `{case_to_show.get('model', 'N/A')}`")
                        st.markdown(f"**Task Type:** `{case_to_show.get('task_type', 'N/A')}`")
                        if pd.notna(case_to_show.get('test_description')): st.markdown(f"**Description:** {case_to_show.get('test_description')}")
                        st.markdown("---"); st.markdown("**Observations:**")
                        obs_text = case_to_show.get('Observations', "_No specific observations generated._")
                        st.markdown(obs_text if obs_text.strip() else "_No specific observations generated._")
                        st.markdown("**Potential Actions:**")
                        act_text = case_to_show.get('Potential Actions', "_No specific automated suggestions._")
                        st.markdown(act_text if act_text.strip() else "_No specific automated suggestions._")
                        st.markdown("**Metrics Not Computed or Not Applicable:**")
                        na_text = case_to_show.get('Metrics Not Computed or Not Applicable', "_All relevant metrics computed or no notes._")
                        st.markdown(na_text if na_text.strip() else "_All relevant metrics computed or no notes._")
                        st.markdown("---")
                        with st.expander("View Question, Ground Truth, and Answer"):
                            st.markdown(f"**Question:**\n```\n{case_to_show.get('question', '')}\n```")
                            st.markdown(f"**Ground Truth:**\n```\n{case_to_show.get('ground_truth', '')}\n```")
                            st.markdown(f"**LLM Answer:**\n```\n{case_to_show.get('answer', '')}\n```")
                            if pd.notna(case_to_show.get('contexts')): st.markdown(f"**Contexts (if provided):**\n```\n{case_to_show.get('contexts')}\n```")
                    else: st.warning(f"Could not find data for ID: {selected_id_for_interp}")
        else: st.info("Run evaluation to generate individual scores with IDs for detailed interpretation.")
        st.divider(); st.subheader("Download Individual Scores Report (with Interpretations)")
        csv_download_df_ind = st.session_state.individual_scores_df.copy() 
        csv_id_cols = ['id', 'task_type', 'model', 'test_description']
        csv_metric_cols = [m_key for m_key in METRIC_INFO.keys() if m_key in csv_download_df_ind.columns] 
        csv_interp_cols = ['Observations', 'Potential Actions', 'Metrics Not Computed or Not Applicable']
        csv_input_output_cols = ['question', 'ground_truth', 'answer', 'ref_facts', 'ref_key_points', 'contexts']
        csv_final_order = []
        for col_group in [csv_id_cols, csv_metric_cols, csv_interp_cols, csv_input_output_cols]:
            for col in col_group:
                if col in csv_download_df_ind.columns and col not in csv_final_order: csv_final_order.append(col)
        csv_remaining_cols = sorted([col for col in csv_download_df_ind.columns if col not in csv_final_order and not col.startswith('_st_')])
        csv_final_order.extend(csv_remaining_cols)
        csv_final_order = [col for col in csv_final_order if col in csv_download_df_ind.columns] 
        csv_data_ind = csv_download_df_ind[csv_final_order].to_csv(index=False, float_format="%.4f").encode('utf-8')
        st.download_button("⬇️ CSV Individual Scores & Interpretations", csv_data_ind, f"individual_eval_scores_interpreted_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv", key="dl_csv_ind_interpreted_results_tab")
    else: st.info("No individual scores to display. Run an evaluation.")


def render_aggregated_results_sub_tab(interpretation_engine): # Pass the engine
    """Renders the 'Aggregated Results' sub-tab within 'Evaluation & Results'."""
    st.markdown("Aggregated view transforms individual data points into insights about the LLM's overall behavior...") 
    if st.session_state.aggregated_results_df is not None and not st.session_state.aggregated_results_df.empty:
        agg_df = st.session_state.aggregated_results_df
        metrics_to_display_non_placeholder = st.session_state.metrics_for_agg_display
        if not metrics_to_display_non_placeholder:
            st.info("No non-placeholder metrics with valid scores to display in aggregated summary...") 
            simple_formatter = {col: "{:.4f}" for col in agg_df.select_dtypes(include=np.number).columns}
            st.dataframe(agg_df.style.format(formatter=simple_formatter, na_rep='NaN'), use_container_width=True)
        else:
            st.markdown("#### 🏆 Best Model Summary (Highlights)")
            st.caption("Top models per task based on key, non-placeholder metrics...") 
            
            key_metrics_per_task = {
                task: [m for m in metrics if not is_placeholder_metric(m)] 
                for task, metrics in KEY_METRICS_PER_TASK_FOR_HIGHLIGHTS.items()
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
                        if not current_task_key_metrics: st.markdown("_No key, non-placeholder metrics for highlights._"); continue
                        for metric_bm in current_task_key_metrics:
                            info_bm = METRIC_INFO.get(metric_bm, {})
                            if not info_bm : continue 
                            higher_better_bm = info_bm.get('higher_is_better', True)
                            valid_scores_df_bm = task_df_bm.dropna(subset=[metric_bm])
                            if valid_scores_df_bm.empty or (valid_scores_df_bm[metric_bm].abs() < 1e-9).all(): continue 
                            best_score_idx_bm = valid_scores_df_bm[metric_bm].idxmax() if higher_better_bm else valid_scores_df_bm[metric_bm].idxmin()
                            best_row_bm = valid_scores_df_bm.loc[best_score_idx_bm]
                            best_model_bm = best_row_bm['model']
                            best_score_val_bm = best_row_bm[metric_bm]
                            if pd.notna(best_score_val_bm) and not np.isclose(best_score_val_bm, 0.0, atol=1e-9): 
                                best_performers_details.append({
                                    "metric_name_display": f"{get_metric_display_name(metric_bm, False)} {get_metric_indicator(metric_bm)}", 
                                    "model": best_model_bm, "score": best_score_val_bm,
                                    "explanation": info_bm.get('explanation', 'N/A')})
                        if not best_performers_details: st.markdown("_No significant highlights determined._")
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
            st.markdown("#### 📊 Overall Summary Table (Aggregated by Task & Model)")
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
                    col_title = get_metric_display_name(original_metric_key, include_placeholder_tag=False) 
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

            st.markdown("#### 🔍 Interpreting Your Aggregated Results (Experimental)")
            with st.expander("💡 Interpreting Your Aggregated Results (Experimental)", expanded=False):
                st.markdown("This section offers a general interpretation of aggregated scores for non-placeholder metrics...") 
                if agg_df is not None and not agg_df.empty:
                    for task_type_interp_agg in agg_df['task_type'].unique():
                        st.markdown(f"#### Task: {task_type_interp_agg}")
                        task_data_interp_agg = agg_df[agg_df['task_type'] == task_type_interp_agg]
                        for model_name_interp_agg in task_data_interp_agg['model'].unique():
                            model_scores_interp_agg = task_data_interp_agg[task_data_interp_agg['model'] == model_name_interp_agg].iloc[0]
                            st.markdown(f"**Model: `{model_name_interp_agg}`**")
                            
                            # Correctly unpack three values
                            interpretations_agg, suggestions_agg, not_applicable_agg_text = interpretation_engine.generate_aggregated_interpretations(model_scores_interp_agg, task_type_interp_agg)

                            if interpretations_agg and interpretations_agg.strip() and interpretations_agg != "No specific aggregated observations based on current thresholds.":
                                st.markdown("**Observations:**")
                                st.markdown(interpretations_agg) # Already formatted with newlines
                            if suggestions_agg and suggestions_agg.strip() and suggestions_agg != "No specific aggregated suggestions.":
                                st.markdown("**Summary / Potential Actions:**")
                                st.markdown(suggestions_agg) # Already formatted with newlines
                            
                            # Optionally display not_applicable_agg_text if it contains meaningful info
                            if not_applicable_agg_text and not_applicable_agg_text.strip() and not_applicable_agg_text != "All relevant aggregated metrics computed or no specific non-applicability notes.":
                                st.markdown("**Notes on Aggregated Metrics:**")
                                st.markdown(not_applicable_agg_text)

                            if (not interpretations_agg or interpretations_agg == "No specific aggregated observations based on current thresholds.") and \
                               (not suggestions_agg or suggestions_agg == "No specific aggregated suggestions."):
                                st.markdown("_No specific interpretations or suggestions generated for this model/task combination based on available aggregated scores._")
                            st.markdown("---")
                else: st.info("Run an evaluation to see interpretations.")
            st.markdown("---")

            st.markdown("#### 📊 Task Specific Metric Table & Chart 📈  ")
            available_tasks_agg = sorted(agg_df['task_type'].unique()) if 'task_type' in agg_df else []
            if not available_tasks_agg: st.info("No tasks found in aggregated results.")
            else:
                task_tabs_agg = st.tabs([f"Task: {task}" for task in available_tasks_agg])
                for i_task_tab, task_type_tab in enumerate(available_tasks_agg):
                    with task_tabs_agg[i_task_tab]:
                        task_df_agg = agg_df[agg_df['task_type'] == task_type_tab].copy()
                        task_specific_metrics_for_task_dim_view = [
                            m for m in get_metrics_for_task(task_type_tab) 
                            if m in agg_df.columns and m in metrics_to_display_non_placeholder 
                        ]
                        if not task_specific_metrics_for_task_dim_view: st.info(f"No relevant, non-placeholder metrics with valid scores for task '{task_type_tab}'."); continue
                        relevant_categories_agg = sorted(list(set(METRIC_INFO[m]['category'] for m in task_specific_metrics_for_task_dim_view if m in METRIC_INFO)))
                        ordered_relevant_categories_agg = [cat for cat in CATEGORY_ORDER if cat in relevant_categories_agg]
                        if not ordered_relevant_categories_agg: st.info(f"No metric categories with displayable metrics for task '{task_type_tab}'."); continue
                        dimension_tabs_agg = st.tabs([f"{cat}" for cat in ordered_relevant_categories_agg])
                        for j_dim, category in enumerate(ordered_relevant_categories_agg):
                            with dimension_tabs_agg[j_dim]:
                                metrics_in_category_task_agg = [m for m in task_specific_metrics_for_task_dim_view if METRIC_INFO.get(m, {}).get('category') == category]
                                if not metrics_in_category_task_agg : st.write(f"_No '{category}' metrics available or selected for display._"); continue
                                cols_to_show_agg_dim = ['model', 'num_samples'] + metrics_in_category_task_agg
                                cols_to_show_present_agg_dim = [c for c in cols_to_show_agg_dim if c in task_df_agg.columns]
                                st.markdown(f"###### {category} Metrics Table (Aggregated for Task: {task_type_tab})")
                                filtered_df_dim_agg = task_df_agg[cols_to_show_present_agg_dim].copy()
                                new_cat_columns_agg = {}
                                for col_key_dim in filtered_df_dim_agg.columns:
                                    if col_key_dim in metrics_in_category_task_agg: 
                                        indicator = get_metric_indicator(col_key_dim)
                                        col_title = get_metric_display_name(col_key_dim, include_placeholder_tag=False) 
                                        new_cat_columns_agg[col_key_dim] = f"{col_title} {indicator}".strip()
                                    elif col_key_dim in ['model', 'num_samples']: new_cat_columns_agg[col_key_dim] = col_key_dim.replace('_', ' ').title()
                                filtered_df_dim_agg.rename(columns=new_cat_columns_agg, inplace=True)
                                display_dim_cols_agg = [new_cat_columns_agg.get(col,col) for col in cols_to_show_present_agg_dim if new_cat_columns_agg.get(col,col) in filtered_df_dim_agg.columns]
                                st.dataframe(filtered_df_dim_agg[display_dim_cols_agg].style.pipe(apply_color_gradient, METRIC_INFO), use_container_width=True)
                                st.markdown(f"###### {category} Charts (Aggregated for Task: {task_type_tab})")
                                plottable_metrics_agg = [m for m in metrics_in_category_task_agg if pd.api.types.is_numeric_dtype(task_df_agg[m])]
                                if not plottable_metrics_agg: st.info("No numeric metrics in this category for charting.")
                                else:
                                    metric_display_options_agg = {f"{get_metric_display_name(m, False)} {get_metric_indicator(m)}".strip(): m for m in plottable_metrics_agg}
                                    selected_metric_display_agg = st.selectbox(
                                        f"Metric for {task_type_tab} - {category}:", list(metric_display_options_agg.keys()),
                                        key=f"chart_sel_agg_{task_type_tab}_{category.replace(' ','_').replace('/','_')}_{i_task_tab}_{j_dim}" 
                                    )
                                    if selected_metric_display_agg:
                                        selected_metric_chart_agg = metric_display_options_agg[selected_metric_display_agg]
                                        metric_explanation_agg = METRIC_INFO.get(selected_metric_chart_agg, {}).get('explanation', "N/A")
                                        st.caption(f"**Definition ({get_metric_display_name(selected_metric_chart_agg, False)}):** {metric_explanation_agg}")
                                        try:
                                            fig_agg = px.bar(task_df_agg, x='model', y=selected_metric_chart_agg, title=f"{selected_metric_display_agg} Scores",
                                                        labels={'model': 'Model / Config', selected_metric_chart_agg: selected_metric_display_agg},
                                                        color='model', text_auto='.4f')
                                            fig_agg.update_layout(xaxis_title="Model / Config", yaxis_title=selected_metric_display_agg); fig_agg.update_traces(textposition='outside')
                                            st.plotly_chart(fig_agg, use_container_width=True)
                                        except Exception as e_chart: st.error(f"Chart error for {selected_metric_display_agg}: {e_chart}")
            st.divider()
            st.subheader("Download Aggregated Reports")
            if agg_df is not None and not agg_df.empty:
                col1_agg_dl, col2_agg_dl = st.columns(2)
                csv_data_agg = agg_df.to_csv(index=False, float_format="%.4f").encode('utf-8') 
                md_content_agg = f"# LLM Evaluation Aggregated Report ({datetime.datetime.now():%Y-%m-%d %H:%M})\n\n"
                agg_df_md_display_dl = agg_df.copy() # Use the original agg_df for markdown content generation
                
                # For Markdown table (numerical part)
                md_content_agg += "## Overall Performance Summary (Aggregated Numerical Scores)\n\n"
                agg_df_md_numeric_part = agg_df.drop(columns=['Aggregated Observations', 'Aggregated Potential Actions', 'Aggregated Metrics Not Computed'], errors='ignore')
                
                renamed_cols_md_numeric_dl = {}
                display_cols_for_md_numeric_table = []
                static_cols_display_dl_numeric = ['task_type', 'model', 'num_samples'] 

                for col_md_static_num in static_cols_display_dl_numeric:
                    if col_md_static_num in agg_df_md_numeric_part.columns:
                        renamed_name_num = col_md_static_num.replace('_', ' ').title()
                        renamed_cols_md_numeric_dl[col_md_static_num] = renamed_name_num
                        display_cols_for_md_numeric_table.append(renamed_name_num)
                
                sorted_metric_keys_for_md_numeric = sorted(
                    [m for m in agg_df_md_numeric_part.columns if m in METRIC_INFO],
                    key=lambda m_key: (is_placeholder_metric(m_key), METRIC_INFO[m_key]['category'], METRIC_INFO[m_key]['name'])
                )
                for m_key_md_num in sorted_metric_keys_for_md_numeric:
                    renamed_m_key_md_num = get_metric_display_name(m_key_md_num, include_placeholder_tag=True) + \
                                           (f" {get_metric_indicator(m_key_md_num)}" if not is_placeholder_metric(m_key_md_num) else "")
                    renamed_cols_md_numeric_dl[m_key_md_num] = renamed_m_key_md_num.strip()
                    if renamed_m_key_md_num.strip() not in display_cols_for_md_numeric_table : # Check if already added
                         display_cols_for_md_numeric_table.append(renamed_m_key_md_num.strip())


                agg_df_md_numeric_part.rename(columns=renamed_cols_md_numeric_dl, inplace=True)
                display_cols_for_md_numeric_table = [col for col in display_cols_for_md_numeric_table if col in agg_df_md_numeric_part.columns] # Ensure columns exist
                md_content_agg += agg_df_md_numeric_part[display_cols_for_md_numeric_table].to_markdown(index=False, floatfmt=".4f")
                md_content_agg += "\n\n"

                # Add separate section for aggregated textual interpretations in Markdown
                if any(col in agg_df.columns for col in ['Aggregated Observations', 'Aggregated Potential Actions', 'Aggregated Metrics Not Computed']):
                    md_content_agg += "## Aggregated Interpretations per Task/Model\n\n"
                    for index, agg_row_md_interp in agg_df.iterrows():
                        md_content_agg += f"### Task: `{agg_row_md_interp.get('task_type', 'N/A')}`, Model: `{agg_row_md_interp.get('model', 'N/A')}`\n\n"
                        if 'Aggregated Observations' in agg_row_md_interp and pd.notna(agg_row_md_interp['Aggregated Observations']):
                            md_content_agg += "**Observations (Aggregated):**\n"
                            md_content_agg += f"{agg_row_md_interp['Aggregated Observations']}\n\n"
                        if 'Aggregated Potential Actions' in agg_row_md_interp and pd.notna(agg_row_md_interp['Aggregated Potential Actions']):
                            md_content_agg += "**Potential Actions (Aggregated):**\n"
                            md_content_agg += f"{agg_row_md_interp['Aggregated Potential Actions']}\n\n"
                        if 'Aggregated Metrics Not Computed' in agg_row_md_interp and pd.notna(agg_row_md_interp['Aggregated Metrics Not Computed']):
                            md_content_agg += "**Metrics Not Computed/Applicable (Aggregated):**\n"
                            md_content_agg += f"{agg_row_md_interp['Aggregated Metrics Not Computed']}\n\n"
                        md_content_agg += "---\n\n"

                md_content_agg += "_End of Aggregated Summary_"
                
                with col1_agg_dl: st.download_button("⬇️ CSV Aggregated Results (with Interpretations)", csv_data_agg, f"aggregated_eval_results_interpreted_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv", key="dl_csv_agg_results_tab_interp")
                with col2_agg_dl: st.download_button("⬇️ MD Aggregated Summary (with Interpretations)", md_content_agg.encode('utf-8'), f"aggregated_eval_summary_interpreted_{datetime.datetime.now():%Y%m%d_%H%M%S}.md", "text/markdown", key="dl_md_agg_results_tab_interp")
    else: st.info("No aggregated results to display. Run an evaluation.")

