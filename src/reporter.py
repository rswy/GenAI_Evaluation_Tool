# src/reporter.py
import pandas as pd
from pathlib import Path
import datetime

# METRIC_INFO might still be useful for indicating higher_is_better for aggregated views
METRIC_INFO = {
    "bleu": {"higher_is_better": True}, "rouge_1": {"higher_is_better": True},
    "rouge_2": {"higher_is_better": True}, "rouge_l": {"higher_is_better": True},
    "meteor": {"higher_is_better": True}, "accuracy": {"higher_is_better": True},
    "precision": {"higher_is_better": True}, "recall": {"higher_is_better": True},
    "f1_score": {"higher_is_better": True},
    "fact_presence_score": {"higher_is_better": True},
    "completeness_score": {"higher_is_better": True},
    "length_ratio": {"higher_is_better": False}, # Closer to 1 is often better, but False implies lower is generally preferred if not 1.
    "safety_keyword_score": {"higher_is_better": True}, # Higher means fewer keywords
    "pii_detection_score": {"higher_is_better": True}  # Higher means fewer PII patterns
    # Add other metrics here if they appear in aggregated results
}

def generate_report(individual_scores_df, aggregated_scores_df, output_dir="reports"):
    """
    Generates evaluation reports from both individual and aggregated results.
    'contexts' field has been removed from consideration in column ordering.

    Args:
        individual_scores_df (pd.DataFrame): DataFrame with scores for each test case.
        aggregated_scores_df (pd.DataFrame): DataFrame with aggregated scores per task/model.
        output_dir (str or Path): Directory to save the reports.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- 1. Detailed Individual Scores (CSV) ---
    if individual_scores_df is not None and not individual_scores_df.empty:
        detailed_csv_path = output_dir_path / f"evaluation_report_individual_scores_{timestamp}.csv"
        print(f"\n--- Generating Detailed Individual Scores (CSV) ---")
        try:
            # Reorder columns for better readability: original inputs first, then metrics
            # 'contexts' removed from this list
            original_input_cols = ['id', 'task_type', 'model', 'question', 'ground_truth', 'answer', 'ref_facts', 'ref_key_points', 'test_description']
            present_original_cols = [col for col in original_input_cols if col in individual_scores_df.columns]
            
            # Identify metric columns - any column not in present_original_cols and not a special/internal column
            # This assumes that all other columns are metric scores.
            # If there are other non-metric columns, they might need explicit exclusion.
            metric_cols = sorted([col for col in individual_scores_df.columns if col not in present_original_cols])
            
            # Ensure 'id' is first if present
            final_order = []
            if 'id' in present_original_cols:
                final_order.append('id')
                present_original_cols.remove('id')
            
            final_order.extend(present_original_cols)
            final_order.extend(metric_cols)
            
            # Filter out columns not present in DataFrame to avoid KeyError (though they should be if logic is correct)
            final_order = [col for col in final_order if col in individual_scores_df.columns]


            individual_scores_df[final_order].to_csv(detailed_csv_path, index=False, float_format="%.4f")
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
                
                # Create a display version for Markdown with indicators if METRIC_INFO is comprehensive
                agg_display_df = aggregated_scores_df.copy()
                renamed_cols = {}
                for col in agg_display_df.columns:
                    metric_info_entry = METRIC_INFO.get(col)
                    if metric_info_entry: # Check if the column is a known metric
                        indicator = "⬆️" if metric_info_entry.get("higher_is_better", True) else "⬇️" # Default to True if not specified
                        renamed_cols[col] = f"{col} {indicator}"
                agg_display_df.rename(columns=renamed_cols, inplace=True)

                f.write(agg_display_df.to_markdown(index=False, floatfmt=".4f"))
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
            aggregated_scores_df.to_csv(agg_csv_path, index=False, float_format="%.4f")
            print(f"Aggregated CSV report saved to: {agg_csv_path}")
        except Exception as e:
            print(f"Error generating aggregated CSV report: {e}")
            import traceback
            print(traceback.format_exc())
    else:
        print("No aggregated scores to report or DataFrame is empty.")