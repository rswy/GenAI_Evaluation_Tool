# main.py
import argparse
import sys
from pathlib import Path
import json # Needed for loading JSON directly

# Ensure the src directory is in the Python path
project_root = Path(__file__).resolve().parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
     sys.path.insert(0, str(src_path))

try:
    from src.data_loader import load_data
    from src.evaluator import evaluate_model_responses # evaluate_model_responses will now return two DataFrames
    from src.reporter import generate_report # generate_report will now accept two DataFrames
    from src.file_converter import convert_excel_to_data, convert_csv_to_data
    from src.mock_data_generator import generate_mock_data_flat, save_mock_data
    import tasks.task_registry # Ensure tasks are discoverable
    import src.metrics # Import the package to ensure it's discoverable
except ImportError as e:
    print(f"Error importing framework modules: {e}")
    print(f"Current sys.path: {sys.path}")
    # ... (rest of error message)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="LLM Evaluation Framework CLI. Run evaluation from file or generate mock data.",
        formatter_class=argparse.RawTextHelpFormatter
        )
    # ... (argparse setup remains the same) ...
    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        '--generate-mock-data',
        action='store_true',
        help="Generate mock data ('data/llm_eval_mock_data.json' and .csv) and exit."
    )
    input_group.add_argument(
        "--input-file",
        type=str,
        help="Path to the input data file (JSON, Excel .xlsx, or CSV .csv using the flat format)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to save evaluation reports (used only with --input-file)."
    )
    parser.add_argument(
        "--mock-data-output-base",
        type=str,
        default="data/llm_eval_mock_data", 
        help="Base path and filename for generated mock data (used only with --generate-mock-data)."
    )
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
            print(f"Error generating mock data: {e}")
            sys.exit(1)
        sys.exit(0)

    if args.input_file:
        input_file_path = project_root / args.input_file
        test_cases = None

        if not input_file_path.exists():
            print(f"Error: Input data file not found at {input_file_path}")
            sys.exit(1)

        print(f"Processing data file: {input_file_path}")
        file_suffix = input_file_path.suffix.lower()

        try:
            if file_suffix == ".xlsx":
                print("Excel file detected. Converting...")
                test_cases = convert_excel_to_data(input_file_path)
            elif file_suffix == ".csv":
                print("CSV file detected. Converting...")
                test_cases = convert_csv_to_data(input_file_path)
            elif file_suffix == ".json":
                print("JSON file detected. Loading...")
                test_cases = load_data(input_file_path) 
            else:
                print(f"Error: Unsupported file format ('{file_suffix}'). Use .json, .xlsx, or .csv.")
                sys.exit(1)

            if test_cases is None:
                 print("Failed to load or convert data. Exiting.")
                 sys.exit(1)
        except Exception as e:
             print(f"An error occurred during data loading/conversion: {e}")
             import traceback
             print(traceback.format_exc())
             sys.exit(1)

        if not test_cases:
            print("No valid test cases found after loading/conversion. Exiting.")
            sys.exit(1)
        print(f"Loaded/Converted {len(test_cases)} test cases.")
        print("-" * 30)

        print("Starting evaluation...")
        individual_scores_df, aggregated_scores_df = pd.DataFrame(), pd.DataFrame() # Initialize
        try:
            # evaluate_model_responses now returns two DataFrames
            individual_scores_df, aggregated_scores_df = evaluate_model_responses(test_cases)
            
            if individual_scores_df.empty and aggregated_scores_df.empty:
                print("Evaluation did not produce any results (both individual and aggregated DataFrames are empty).")
            else:
                 print("Evaluation successful.")
                 # print("Aggregated Results:")
                 # print(aggregated_scores_df.to_string()) # Optional print
                 # print("\nFirst 5 Individual Results:")
                 # print(individual_scores_df.head().to_string()) # Optional print
        except Exception as e:
             print(f"An error occurred during evaluation: {e}")
             import traceback
             print(traceback.format_exc())
             sys.exit(1)

        print("-" * 30)

        output_directory = project_root / args.output_dir
        print(f"Generating reports in: {output_directory}")
        try:
            # Pass both DataFrames to the reporter
            generate_report(individual_scores_df, aggregated_scores_df, output_dir=output_directory)
        except Exception as e:
             print(f"An error occurred during report generation: {e}")
             import traceback
             print(traceback.format_exc())
             sys.exit(1)

        print("-" * 30)
        print("Command-line evaluation complete.")


def json_to_csv():
        
    # You can save this list to a JSON file or convert it to a DataFrame for CSV/Excel
    import json
    import pandas as pd
    # Save to CSV
    try:
        df = pd.read_json("data/hr_uob.json")
        df.to_csv("data/hr_uob.csv", index=False)
        print("Saved test cases to hr_policy_test_cases.json and hr_policy_test_cases.csv")
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        print("Saved test cases only to hr_policy_test_cases.json")


if __name__ == "__main__":
    main()
    # json_to_csv()