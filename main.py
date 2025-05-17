# main.py
import argparse
import sys
from pathlib import Path
import pandas as pd 
import traceback 

project_root = Path(__file__).resolve().parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(src_path))

try:
    from src.data_loader import load_data
    from src.evaluator import evaluate_model_responses
    from src.reporter import generate_report
    from src.file_converter import convert_excel_to_data, convert_csv_to_data
    from src.mock_data_generator import generate_mock_data_flat, save_mock_data
    from src.interpretation_engine import generate_single_case_interpretation, generate_aggregated_interpretations # Import new function
    import src.tasks.task_registry 
    import src.metrics 
except ImportError as e:
    print(f"Error importing framework modules: {e}")
    print(f"Current sys.path: {sys.path}")
    print("Please ensure that the 'src' directory is correctly added to your Python path,")
    print("and all necessary __init__.py files are present in 'src' and its subdirectories.")
    print(f"Traceback: {traceback.format_exc()}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="LLM Evaluation Framework CLI. Run evaluation from file or generate mock data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--generate-mock-data', action='store_true', help="Generate mock data and exit.")
    input_group.add_argument("--input-file", type=str, help="Path to the input data file (JSON, Excel .xlsx, or CSV .csv).")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory to save evaluation reports.")
    parser.add_argument("--mock-data-output-base", type=str, default="data/llm_eval_mock_data_cli", help="Base path for generated mock data.")
    args = parser.parse_args()

    if args.generate_mock_data:
        print("Generating mock data...")
        mock_data_base_path = project_root / args.mock_data_output_base
        mock_data_base_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            mock_data_list = generate_mock_data_flat(num_samples_per_task=3)
            save_mock_data(mock_data_list, output_dir=mock_data_base_path.parent, base_filename=mock_data_base_path.name)
            print(f"Mock data generated: {mock_data_base_path}.json / .csv")
        except Exception as e:
            print(f"Error generating mock data: {e}\n{traceback.format_exc()}"); sys.exit(1)
        sys.exit(0)

    if args.input_file:
        input_file_path = project_root / args.input_file
        test_cases = None
        if not input_file_path.exists(): print(f"Error: Input data file not found at {input_file_path}"); sys.exit(1)

        print(f"Processing data file: {input_file_path}")
        file_suffix = input_file_path.suffix.lower()
        try:
            if file_suffix == ".xlsx": test_cases = convert_excel_to_data(input_file_path)
            elif file_suffix == ".csv": test_cases = convert_csv_to_data(input_file_path)
            elif file_suffix == ".json": test_cases = load_data(input_file_path)
            else: print(f"Error: Unsupported file format ('{file_suffix}')."); sys.exit(1)
            if test_cases is None: print("Failed to load or convert data. Exiting."); sys.exit(1)
        except Exception as e: print(f"Error during data loading: {e}\n{traceback.format_exc()}"); sys.exit(1)

        if not test_cases: print("No valid test cases found. Exiting."); sys.exit(1)
        print(f"Loaded/Converted {len(test_cases)} test cases.\n" + "-" * 30)

        print("Starting evaluation...")
        individual_scores_df, aggregated_scores_df = pd.DataFrame(), pd.DataFrame()
        try:
            individual_scores_df, aggregated_scores_df = evaluate_model_responses(test_cases)
            if individual_scores_df.empty and aggregated_scores_df.empty:
                print("Evaluation did not produce any results.")
            else:
                print("Evaluation successful. Generating interpretations...")
                if individual_scores_df is not None and not individual_scores_df.empty:
                    print("Applying interpretation logic to individual scores...")
                    try:
                        interpretations_output_ind = individual_scores_df.apply(
                            lambda row: generate_single_case_interpretation(row, row.get('task_type')), axis=1
                        )
                        individual_scores_df['Observations'] = interpretations_output_ind.apply(lambda x: x[0])
                        individual_scores_df['Potential Actions'] = interpretations_output_ind.apply(lambda x: x[1])
                        individual_scores_df['Metrics Not Computed or Not Applicable'] = interpretations_output_ind.apply(lambda x: x[2])
                        print("Individual interpretations added.")
                    except Exception as e_interp_ind:
                        print(f"Error during individual interpretation generation: {e_interp_ind}\n{traceback.format_exc()}")
                        print("Proceeding without full individual interpretation columns.")
                
                # --- Add Interpretation Columns to Aggregated Scores DataFrame ---
                if aggregated_scores_df is not None and not aggregated_scores_df.empty:
                    print("Applying interpretation logic to aggregated scores...")
                    agg_obs_list, agg_actions_list, agg_na_list = [], [], []
                    try:
                        for index, agg_row in aggregated_scores_df.iterrows():
                            task_type_agg = agg_row.get('task_type')
                            # Pass the entire row (which is a Series of mean scores)
                            obs, act, na = generate_aggregated_interpretations(agg_row, task_type_agg)
                            agg_obs_list.append(obs)
                            agg_actions_list.append(act)
                            agg_na_list.append(na)
                        
                        aggregated_scores_df['Aggregated Observations'] = agg_obs_list
                        aggregated_scores_df['Aggregated Potential Actions'] = agg_actions_list
                        aggregated_scores_df['Aggregated Metrics Not Computed'] = agg_na_list
                        print("Aggregated interpretations added.")
                    except Exception as e_interp_agg:
                        print(f"Error during aggregated interpretation generation: {e_interp_agg}\n{traceback.format_exc()}")
                        print("Proceeding without full aggregated interpretation columns.")

        except Exception as e: print(f"Error during evaluation: {e}\n{traceback.format_exc()}"); sys.exit(1)
        
        print("-" * 30)
        output_directory = project_root / args.output_dir
        print(f"Generating reports in: {output_directory}")
        try:
            generate_report(individual_scores_df, aggregated_scores_df, output_dir=output_directory)
            print(f"Reports successfully generated in {output_directory.resolve()}")
        except Exception as e: print(f"Error during report generation: {e}\n{traceback.format_exc()}"); sys.exit(1)
        
        print("-" * 30 + "\nCommand-line evaluation complete.")

if __name__ == "__main__":
    main()
