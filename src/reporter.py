# src/reporter.py
import pandas as pd
from pathlib import Path
import datetime

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
    "completeness_score": {"higher_is_better": True, "name": "Key Point Coverage"},
    "length_ratio": {"higher_is_better": False, "name": "Length Ratio"}, 
    "safety_keyword_score": {"higher_is_better": True, "name": "Safety Keyword Score"}, 
    "pii_detection_score": {"higher_is_better": True, "name": "PII Detection Score"},
    "professional_tone_score": {"higher_is_better": True, "status": "placeholder", "name": "Professional Tone"},
    "refusal_quality_score": {"higher_is_better": True, "status": "placeholder", "name": "Refusal Quality"},
    "nli_entailment_score": {"higher_is_better": True, "status": "placeholder", "name": "NLI Entailment"},
    "llm_judge_factuality": {"higher_is_better": True, "status": "placeholder", "name": "LLM Judge Factuality"}
}

IND_INTERPRETATION_COLS = ['Observations', 'Potential Actions', 'Metrics Not Computed or Not Applicable']
AGG_INTERPRETATION_COLS = ['Aggregated Observations', 'Aggregated Potential Actions', 'Aggregated Metrics Not Computed']


def get_metric_display_name_reporter(metric_key, include_placeholder_tag=True):
    info = METRIC_INFO_REPORTER.get(metric_key, {})
    name = info.get('name', metric_key.replace('_', ' ').title())
    if include_placeholder_tag and info.get("status") == "placeholder": 
        if "(Placeholder)" not in name: name += " (Placeholder)"
    return name

def get_metric_indicator_reporter(metric_key):
    info = METRIC_INFO_REPORTER.get(metric_key)
    if not info or info.get("status") == "placeholder": return ""
    return "⬆️" if info.get("higher_is_better", True) else "⬇️"

def generate_report(individual_scores_df, aggregated_scores_df, output_dir="reports"):
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- 1. Detailed Individual Scores (CSV) ---
    if individual_scores_df is not None and not individual_scores_df.empty:
        detailed_csv_path = output_dir_path / f"evaluation_report_individual_scores_{timestamp}.csv"
        print(f"\n--- Generating Detailed Individual Scores (CSV) ---")
        try:
            id_cols = ['id', 'task_type', 'model', 'test_description']
            input_output_cols = ['question', 'ground_truth', 'answer', 'ref_facts', 'ref_key_points', 'contexts']
            metric_keys_in_df = [m_key for m_key in METRIC_INFO_REPORTER.keys() if m_key in individual_scores_df.columns]
            sorted_metric_keys = sorted(
                metric_keys_in_df,
                key=lambda k: (METRIC_INFO_REPORTER.get(k, {}).get("status") == "placeholder", 
                               list(METRIC_INFO_REPORTER.keys()).index(k) if k in METRIC_INFO_REPORTER else float('inf'))
            )
            present_ind_interpretation_cols = [col for col in IND_INTERPRETATION_COLS if col in individual_scores_df.columns]
            final_order_csv_ind = []
            for group in [id_cols, input_output_cols, sorted_metric_keys, present_ind_interpretation_cols]:
                for col in group:
                    if col in individual_scores_df.columns and col not in final_order_csv_ind:
                        final_order_csv_ind.append(col)
            remaining_cols_ind = [col for col in individual_scores_df.columns if col not in final_order_csv_ind]
            final_order_csv_ind.extend(sorted(remaining_cols_ind))
            final_order_csv_ind = [col for col in final_order_csv_ind if col in individual_scores_df.columns]
            individual_scores_df[final_order_csv_ind].to_csv(detailed_csv_path, index=False, float_format="%.4f")
            print(f"Detailed individual scores CSV saved to: {detailed_csv_path}")
        except Exception as e:
            print(f"Error generating detailed individual scores CSV: {e}\n{traceback.format_exc()}")
    else:
        print("No individual scores to report or DataFrame is empty.")

    # --- 2. Aggregated Summary Report (Markdown and CSV) ---
    if aggregated_scores_df is not None and not aggregated_scores_df.empty:
        agg_md_path = output_dir_path / f"evaluation_report_aggregated_summary_{timestamp}.md"
        agg_csv_path = output_dir_path / f"evaluation_report_aggregated_summary_{timestamp}.csv"
        
        # --- Aggregated CSV (with new interpretation columns) ---
        print(f"\n--- Generating Aggregated Summary (CSV) ---")
        try:
            csv_agg_cols_ordered = []
            static_agg_cols = ['task_type', 'model', 'num_samples']
            for col_csv_static in static_agg_cols:
                 if col_csv_static in aggregated_scores_df.columns:
                      csv_agg_cols_ordered.append(col_csv_static)
            
            metric_cols_for_csv_agg = [m for m in METRIC_INFO_REPORTER.keys() if m in aggregated_scores_df.columns]
            sorted_metric_cols_for_csv_agg = sorted(
                metric_cols_for_csv_agg,
                 key=lambda m_key: (METRIC_INFO_REPORTER.get(m_key, {}).get("status") == "placeholder", 
                                   METRIC_INFO_REPORTER.get(m_key, {}).get("name", m_key))
            )
            csv_agg_cols_ordered.extend(sorted_metric_cols_for_csv_agg)
            
            # Add aggregated interpretation columns
            present_agg_interpretation_cols = [col for col in AGG_INTERPRETATION_COLS if col in aggregated_scores_df.columns]
            csv_agg_cols_ordered.extend(present_agg_interpretation_cols)

            remaining_cols_agg_csv = [c for c in aggregated_scores_df.columns if c not in csv_agg_cols_ordered]
            csv_agg_cols_ordered.extend(sorted(remaining_cols_agg_csv))
            csv_agg_cols_ordered = [col for col in csv_agg_cols_ordered if col in aggregated_scores_df.columns]

            aggregated_scores_df[csv_agg_cols_ordered].to_csv(agg_csv_path, index=False, float_format="%.4f")
            print(f"Aggregated CSV report saved to: {agg_csv_path}")
        except Exception as e:
            print(f"Error generating aggregated CSV report: {e}\n{traceback.format_exc()}")

        # --- Aggregated Markdown ---
        print(f"\n--- Generating Aggregated Summary Report (Markdown) ---")
        try:
            with open(agg_md_path, "w", encoding="utf-8") as f:
                f.write(f"# LLM Evaluation Aggregated Summary\n\n")
                f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("## Overall Performance Summary (Aggregated Numerical Scores)\n\n")
                
                # For Markdown table, only include numerical metrics and static info for brevity
                agg_display_df_md_numeric = aggregated_scores_df.copy()
                cols_to_drop_for_md_table = AGG_INTERPRETATION_COLS # Drop textual interpretations from this table
                agg_display_df_md_numeric = agg_display_df_md_numeric.drop(columns=cols_to_drop_for_md_table, errors='ignore')

                renamed_cols_md_numeric = {}
                display_cols_md_numeric_final = []
                static_cols_md_numeric = ['task_type', 'model', 'num_samples']

                for col_md_static in static_cols_md_numeric:
                    if col_md_static in agg_display_df_md_numeric.columns:
                        renamed_name = col_md_static.replace('_', ' ').title()
                        renamed_cols_md_numeric[col_md_static] = renamed_name
                        display_cols_md_numeric_final.append(renamed_name)
                
                metric_cols_in_agg_df_numeric = [m for m in METRIC_INFO_REPORTER.keys() if m in agg_display_df_md_numeric.columns]
                sorted_metric_cols_agg_numeric = sorted(
                    metric_cols_in_agg_df_numeric,
                    key=lambda m_key: (METRIC_INFO_REPORTER.get(m_key, {}).get("status") == "placeholder", 
                                       METRIC_INFO_REPORTER.get(m_key, {}).get("name", m_key)) 
                )

                for m_key_agg_numeric in sorted_metric_cols_agg_numeric:
                    display_name_agg_numeric = get_metric_display_name_reporter(m_key_agg_numeric, include_placeholder_tag=True)
                    indicator_agg_numeric = get_metric_indicator_reporter(m_key_agg_numeric)
                    renamed_cols_md_numeric[m_key_agg_numeric] = f"{display_name_agg_numeric} {indicator_agg_numeric}".strip()
                    display_cols_md_numeric_final.append(renamed_cols_md_numeric[m_key_agg_numeric])

                agg_display_df_md_numeric.rename(columns=renamed_cols_md_numeric, inplace=True)
                display_cols_md_numeric_final = [col for col in display_cols_md_numeric_final if col in agg_display_df_md_numeric.columns]
                f.write(agg_display_df_md_numeric[display_cols_md_numeric_final].to_markdown(index=False, floatfmt=".4f"))
                f.write("\n\n")

                # Add a separate section for aggregated textual interpretations
                if any(col in aggregated_scores_df.columns for col in AGG_INTERPRETATION_COLS):
                    f.write("## Aggregated Interpretations per Task/Model\n\n")
                    for index, agg_row_md in aggregated_scores_df.iterrows():
                        f.write(f"### Task: `{agg_row_md.get('task_type', 'N/A')}`, Model: `{agg_row_md.get('model', 'N/A')}`\n\n")
                        if 'Aggregated Observations' in agg_row_md and pd.notna(agg_row_md['Aggregated Observations']):
                            f.write("**Observations (Aggregated):**\n")
                            f.write(f"{agg_row_md['Aggregated Observations']}\n\n")
                        if 'Aggregated Potential Actions' in agg_row_md and pd.notna(agg_row_md['Aggregated Potential Actions']):
                            f.write("**Potential Actions (Aggregated):**\n")
                            f.write(f"{agg_row_md['Aggregated Potential Actions']}\n\n")
                        if 'Aggregated Metrics Not Computed' in agg_row_md and pd.notna(agg_row_md['Aggregated Metrics Not Computed']):
                            f.write("**Metrics Not Computed/Applicable (Aggregated):**\n")
                            f.write(f"{agg_row_md['Aggregated Metrics Not Computed']}\n\n")
                        f.write("---\n\n")
                
                if individual_scores_df is not None and not individual_scores_df.empty:
                    f.write(f"Detailed individual scores (including textual interpretations) are available in the accompanying CSV: `evaluation_report_individual_scores_{timestamp}.csv`\n")
            print(f"Aggregated Markdown report saved to: {agg_md_path}")
        except Exception as e:
            print(f"Error generating aggregated Markdown report: {e}\n{traceback.format_exc()}")
    else:
        print("No aggregated scores to report or DataFrame is empty.")

if __name__ == '__main__':
    print("Testing reporter functions (CLI version with aggregated interpretations)...")
    dummy_individual_data = {
        'id': ['t1', 't2'], 'task_type': ['rag_faq', 'rag_faq'], 'model': ['modelA', 'modelA'],
        'question': ['Q1', 'Q2'], 'ground_truth': ['GT1', 'GT2'], 'answer': ['A1', 'A2'],
        'bleu': [0.5, 0.6], 'semantic_similarity_score': [0.7, 0.85],
        'fact_presence_score': [1.0, 0.5], 'professional_tone_score': [float('nan'), float('nan')],
        'Observations': ['Ind_Obs1', 'Ind_Obs2'], 
        'Potential Actions': ['Ind_Act1', 'Ind_Act2'],
        'Metrics Not Computed or Not Applicable': ['Ind_NA1', 'Ind_NA2']
    }
    dummy_ind_df = pd.DataFrame(dummy_individual_data)
    
    dummy_aggregated_data = {
        'task_type': ['rag_faq'], 'model': ['modelA'], 'num_samples': [2],
        'bleu': [0.55], 'semantic_similarity_score': [0.775],
        'fact_presence_score': [0.75], 'professional_tone_score': [float('nan')],
        'Aggregated Observations': ["- Agg_Obs1\n- Agg_Obs2"], # Example aggregated text
        'Aggregated Potential Actions': ["- Agg_Act1"],
        'Aggregated Metrics Not Computed': ["- Agg_NA1"]
    }
    dummy_agg_df = pd.DataFrame(dummy_aggregated_data)
    
    test_output_dir = Path("test_cli_reports_output_agg_interp") 
    generate_report(dummy_ind_df, dummy_agg_df, output_dir=test_output_dir)
    print(f"Test CLI reports generated in {test_output_dir.resolve()}")