# src/reporter.py
import pandas as pd
from pathlib import Path
import datetime

# Define METRIC_INFO locally for reporter context, especially for 'higher_is_better' and 'status'
# This should align with METRIC_INFO in streamlit_app.py for consistency in reporting.
# (Ideally, this could be shared from a common config if the project grows more complex)
METRIC_INFO_REPORTER = {
    "bleu": {"higher_is_better": True, "name": "BLEU"}, 
    "rouge_1": {"higher_is_better": True, "name": "ROUGE-1"},
    "rouge_2": {"higher_is_better": True, "name": "ROUGE-2"}, 
    "rouge_l": {"higher_is_better": True, "name": "ROUGE-L"},
    "meteor": {"higher_is_better": True, "name": "METEOR"}, 
    "semantic_similarity_score": {"higher_is_better": True, "name": "Semantic Similarity"}, 
    "accuracy": {"higher_is_better": True, "name": "Accuracy"},
    "precision": {"higher_is_better": True, "name": "Precision"}, 
    "recall": {"higher_is_better": True, "name": "Recall"},
    "f1_score": {"higher_is_better": True, "name": "F1-Score"},
    "fact_presence_score": {"higher_is_better": True, "name": "Fact Presence"},
    "completeness_score": {"higher_is_better": True, "name": "Checklist Completeness"},
    "length_ratio": {"higher_is_better": False, "name": "Length Ratio"}, 
    "safety_keyword_score": {"higher_is_better": True, "name": "Safety Keyword Score"}, 
    "pii_detection_score": {"higher_is_better": True, "name": "PII Detection Score"},
    "professional_tone_score": {"higher_is_better": True, "status": "placeholder", "name": "Professional Tone"},
    "refusal_quality_score": {"higher_is_better": True, "status": "placeholder", "name": "Refusal Quality"},
    "nli_entailment_score": {"higher_is_better": True, "status": "placeholder", "name": "NLI Entailment"},
    "llm_judge_factuality": {"higher_is_better": True, "status": "placeholder", "name": "LLM Judge Factuality"}
}

def get_metric_display_name_reporter(metric_key, include_placeholder_tag=True):
    info = METRIC_INFO_REPORTER.get(metric_key, {})
    name = info.get('name', metric_key.replace('_', ' ').title())
    if include_placeholder_tag and info.get("status") == "placeholder": 
        if "(Placeholder)" not in name: 
            name += " (Placeholder)"
    return name

def get_metric_indicator_reporter(metric_key):
    info = METRIC_INFO_REPORTER.get(metric_key)
    return "⬆️" if info and info.get("higher_is_better", True) else ("⬇️" if info and not info.get("higher_is_better") else "")


def generate_report(individual_scores_df, aggregated_scores_df, output_dir="reports"):
    """
    Generates evaluation reports from both individual and aggregated results.
    Handles new semantic_similarity_score and placeholder statuses.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- 1. Detailed Individual Scores (CSV) ---
    if individual_scores_df is not None and not individual_scores_df.empty:
        detailed_csv_path = output_dir_path / f"evaluation_report_individual_scores_{timestamp}.csv"
        print(f"\n--- Generating Detailed Individual Scores (CSV) ---")
        try:
            # For CSV, use original metric keys for easier machine processing.
            # Order: IDs, inputs, interpretations, then metrics.
            
            id_cols = ['id', 'task_type', 'model', 'test_description']
            input_output_cols = ['question', 'ground_truth', 'answer', 
                                 'ref_facts', 'ref_key_points', 'contexts'] # contexts for legacy
            interpretation_cols = ['Observations', 'Potential Actions', 'Metrics Not Computed or Not Applicable']
            
            # Get all original metric keys present in the DataFrame
            # These are columns that are in METRIC_INFO_REPORTER and also in the df
            metric_keys_in_df = [m_key for m_key in METRIC_INFO_REPORTER.keys() if m_key in individual_scores_df.columns]
            
            # Sort metric keys for consistent ordering, perhaps by a predefined order or alphabetically
            # For now, use the order they appear in METRIC_INFO_REPORTER if available, then others
            sorted_metric_keys = sorted(
                metric_keys_in_df,
                key=lambda k: list(METRIC_INFO_REPORTER.keys()).index(k) if k in METRIC_INFO_REPORTER else float('inf')
            )

            final_order_csv = []
            # Add ID columns
            for col in id_cols:
                if col in individual_scores_df.columns and col not in final_order_csv: final_order_csv.append(col)
            # Add Input/Output columns
            for col in input_output_cols:
                if col in individual_scores_df.columns and col not in final_order_csv: final_order_csv.append(col)
            # Add Interpretation columns
            for col in interpretation_cols:
                if col in individual_scores_df.columns and col not in final_order_csv: final_order_csv.append(col)
            # Add Metric columns (original keys)
            for col in sorted_metric_keys: # These are already filtered to be in df.columns
                if col not in final_order_csv: final_order_csv.append(col)
            
            # Add any remaining columns from the DataFrame that weren't caught
            remaining_cols = [col for col in individual_scores_df.columns if col not in final_order_csv]
            final_order_csv.extend(sorted(remaining_cols))
            
            # Ensure we only select columns that actually exist to prevent KeyErrors
            final_order_csv = [col for col in final_order_csv if col in individual_scores_df.columns]

            individual_scores_df[final_order_csv].to_csv(detailed_csv_path, index=False, float_format="%.4f")
            print(f"Detailed individual scores CSV saved to: {detailed_csv_path}")
        except Exception as e:
            print(f"Error generating detailed individual scores CSV: {e}")
            import traceback
            print(traceback.format_exc())
    else:
        print("No individual scores to report or DataFrame is empty.")

    # --- 2. Aggregated Summary Report (Markdown and CSV) ---
    if aggregated_scores_df is not None and not aggregated_scores_df.empty:
        agg_md_path = output_dir_path / f"evaluation_report_aggregated_summary_{timestamp}.md"
        agg_csv_path = output_dir_path / f"evaluation_report_aggregated_summary_{timestamp}.csv"
        
        print(f"\n--- Generating Aggregated Summary Report (Markdown) ---")
        try:
            with open(agg_md_path, "w", encoding="utf-8") as f:
                f.write(f"# LLM Evaluation Aggregated Summary\n\n")
                f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("## Overall Performance Summary (Aggregated)\n\n")
                
                agg_display_df_md = aggregated_scores_df.copy()
                renamed_cols_md = {}
                
                # Define order for static columns
                static_cols_md = ['task_type', 'model', 'num_samples']
                display_cols_md_final = []

                for col_md_static in static_cols_md:
                    if col_md_static in agg_display_df_md.columns:
                        renamed_name = col_md_static.replace('_', ' ').title()
                        renamed_cols_md[col_md_static] = renamed_name
                        display_cols_md_final.append(renamed_name)
                
                # Get metric columns present in aggregated_df
                metric_cols_in_agg_df = [m for m in METRIC_INFO_REPORTER.keys() if m in agg_display_df_md.columns]
                # Sort them: non-placeholders first, then placeholders
                sorted_metric_cols_agg = sorted(
                    metric_cols_in_agg_df,
                    key=lambda m_key: (METRIC_INFO_REPORTER.get(m_key, {}).get("status") == "placeholder", 
                                       METRIC_INFO_REPORTER.get(m_key, {}).get("name", m_key)) 
                )

                for m_key_agg in sorted_metric_cols_agg:
                    info_agg = METRIC_INFO_REPORTER.get(m_key_agg, {})
                    display_name_agg = get_metric_display_name_reporter(m_key_agg, include_placeholder_tag=True)
                    indicator_agg = get_metric_indicator_reporter(m_key_agg) if info_agg.get("status") != "placeholder" else ""
                    
                    renamed_cols_md[m_key_agg] = f"{display_name_agg} {indicator_agg}".strip()
                    display_cols_md_final.append(renamed_cols_md[m_key_agg])

                agg_display_df_md.rename(columns=renamed_cols_md, inplace=True)
                
                # Ensure only existing columns are selected for the markdown table
                display_cols_md_final = [col for col in display_cols_md_final if col in agg_display_df_md.columns]

                f.write(agg_display_df_md[display_cols_md_final].to_markdown(index=False, floatfmt=".4f"))
                f.write("\n\n")
                if individual_scores_df is not None and not individual_scores_df.empty:
                    f.write(f"Detailed individual scores are available in the accompanying CSV: `evaluation_report_individual_scores_{timestamp}.csv`\n")
            print(f"Aggregated Markdown report saved to: {agg_md_path}")
        except Exception as e:
            print(f"Error generating aggregated Markdown report: {e}")
            import traceback
            print(traceback.format_exc())

        print(f"\n--- Generating Aggregated Summary (CSV) ---")
        try:
            # For CSV, use original column names without indicators for machine readability
            # Order: static cols, then sorted original metric keys
            csv_agg_cols_ordered = []
            for col_csv_static in ['task_type', 'model', 'num_samples']:
                 if col_csv_static in aggregated_scores_df.columns:
                      csv_agg_cols_ordered.append(col_csv_static)
            
            metric_cols_for_csv_agg = [m for m in METRIC_INFO_REPORTER.keys() if m in aggregated_scores_df.columns]
            sorted_metric_cols_for_csv_agg = sorted(
                metric_cols_for_csv_agg,
                 key=lambda m_key: (METRIC_INFO_REPORTER.get(m_key, {}).get("status") == "placeholder", 
                                   METRIC_INFO_REPORTER.get(m_key, {}).get("name", m_key))
            )
            csv_agg_cols_ordered.extend(sorted_metric_cols_for_csv_agg)

            # Add any other columns not caught
            remaining_cols_agg_csv = [c for c in aggregated_scores_df.columns if c not in csv_agg_cols_ordered]
            csv_agg_cols_ordered.extend(sorted(remaining_cols_agg_csv))
            
            csv_agg_cols_ordered = [col for col in csv_agg_cols_ordered if col in aggregated_scores_df.columns]


            aggregated_scores_df[csv_agg_cols_ordered].to_csv(agg_csv_path, index=False, float_format="%.4f")
            print(f"Aggregated CSV report saved to: {agg_csv_path}")
        except Exception as e:
            print(f"Error generating aggregated CSV report: {e}")
            import traceback
            print(traceback.format_exc())
    else:
        print("No aggregated scores to report or DataFrame is empty.")

if __name__ == '__main__':
    # Basic test for reporter (requires creating dummy DataFrames)
    print("Testing reporter functions...")
    
    # Dummy Individual Scores DataFrame
    dummy_individual_data = {
        'id': ['t1', 't2'],
        'task_type': ['rag_faq', 'rag_faq'],
        'model': ['modelA', 'modelA'],
        'question': ['Q1', 'Q2'],
        'ground_truth': ['GT1', 'GT2'],
        'answer': ['A1', 'A2'],
        'bleu': [0.5, 0.6],
        'semantic_similarity_score': [0.7, 0.85],
        'fact_presence_score': [1.0, 0.5],
        'professional_tone_score': [float('nan'), float('nan')], # Placeholder
        'Observations': ['Obs1', 'Obs2'],
        'Potential Actions': ['Act1', 'Act2'],
        'Metrics Not Computed or Not Applicable': ['', 'NLI: Placeholder']
    }
    dummy_ind_df = pd.DataFrame(dummy_individual_data)

    # Dummy Aggregated Scores DataFrame
    dummy_aggregated_data = {
        'task_type': ['rag_faq'],
        'model': ['modelA'],
        'num_samples': [2],
        'bleu': [0.55],
        'semantic_similarity_score': [0.775],
        'fact_presence_score': [0.75],
        'professional_tone_score': [float('nan')]
    }
    dummy_agg_df = pd.DataFrame(dummy_aggregated_data)

    test_output_dir = Path("test_reports_output")
    generate_report(dummy_ind_df, dummy_agg_df, output_dir=test_output_dir)
    print(f"Test reports generated in {test_output_dir.resolve()}")