# src/evaluator.py
import pandas as pd
from tasks.task_registry import (get_metrics_for_task, get_primary_reference_col,
                                 get_primary_prediction_col, CUSTOM_METRIC_KWARG_MAP)
from metrics import get_metric_instances
import warnings
from collections import defaultdict
import traceback
import time

def evaluate_model_responses(test_cases_list):
    """
    Runs the evaluation pipeline, generating both individual and aggregated scores.
    Assumes metric classes' compute() method now processes a single instance.
    """
    if not isinstance(test_cases_list, list) or not test_cases_list:
        print("Error: Invalid or empty test_cases_list provided to evaluator.")
        return pd.DataFrame(), pd.DataFrame()

    individual_results_list = []
    overall_start_time = time.time()
    print("\nStarting Evaluation (Individual Scores)...")

    for i, test_case in enumerate(test_cases_list):
        case_start_time = time.time()
        task_type = test_case.get('task_type')
        model_name = test_case.get('model')
        case_id = test_case.get('id', f"case_{i}")

        if not task_type or not model_name:
            warnings.warn(f"Skipping test case {case_id}: Missing 'task_type' or 'model'.", RuntimeWarning)
            continue

        # print(f"\n--- Processing Test Case ID: {case_id}, Task: {task_type}, Model: {model_name} ---") # Can be verbose

        metric_names_for_task = get_metrics_for_task(task_type)
        primary_ref_col = get_primary_reference_col(task_type)
        primary_pred_col = get_primary_prediction_col(task_type)

        if not metric_names_for_task or not primary_ref_col or not primary_pred_col:
            warnings.warn(f"Skipping {case_id} ({task_type}/{model_name}): Missing metric names or primary column definitions.", RuntimeWarning)
            continue

        primary_reference = str(test_case.get(primary_ref_col, "")).strip()
        prediction = str(test_case.get(primary_pred_col, "")).strip()

        # Prepare kwargs for custom metrics for this single instance
        instance_metric_kwargs = {}
        custom_metrics_needed = {m for m in metric_names_for_task if m in CUSTOM_METRIC_KWARG_MAP}

        if custom_metrics_needed:
            ref_field_values_for_instance = {}
            for metric_name_key in custom_metrics_needed: # e.g., "fact_presence_score"
                kwarg_map_for_metric = CUSTOM_METRIC_KWARG_MAP.get(metric_name_key, {}) # e.g., {"facts": "ref_facts"}
                for kwarg_name, data_column_name in kwarg_map_for_metric.items(): # e.g., kwarg_name="facts", data_column_name="ref_facts"
                    raw_val = test_case.get(data_column_name)
                    if pd.notna(raw_val) and str(raw_val).strip():
                        # Ensure it's a list of strings
                        ref_field_values_for_instance[kwarg_name] = [item.strip() for item in str(raw_val).split(',') if item.strip()]
                    else:
                        ref_field_values_for_instance[kwarg_name] = [] # Empty list if data missing/empty
            instance_metric_kwargs['reference_field_values'] = ref_field_values_for_instance


        metric_instances = get_metric_instances(metric_names_for_task)
        instance_scores = {}
        metric_times = {}

        for metric_key, metric_calculator in metric_instances.items():
            metric_start_time = time.time()
            try:
                # Pass single reference, single prediction, and instance-specific kwargs
                computed_score_dict = metric_calculator.compute(
                    references=primary_reference,      # Single reference string
                    predictions=prediction,            # Single prediction string
                    **instance_metric_kwargs           # Contains reference_field_values for this instance
                )
                if computed_score_dict:
                    instance_scores.update(computed_score_dict)
                else:
                    warnings.warn(f"Metric group '{metric_key}' compute returned None/empty for {case_id}.", RuntimeWarning)

            except Exception as e:
                print(f"    *** ERROR during compute for metric group '{metric_key}' on case {case_id} ***")
                print(f"        Exception Type: {type(e)} | Args: {e.args}")
                print(f"        Traceback: {traceback.format_exc()}")
                # Assign NaNs for expected keys from this calculator if error
                keys_to_nan = []
                if metric_key == "rouge": keys_to_nan = ["rouge_1", "rouge_2", "rouge_l"]
                elif metric_key == "classification": keys_to_nan = ["accuracy", "precision", "recall", "f1_score"]
                elif metric_key == "fact_presence": keys_to_nan = ["fact_presence_score"]
                elif metric_key == "completeness": keys_to_nan = ["completeness_score"]
                elif metric_key == "length_ratio": keys_to_nan = ["length_ratio"]
                elif metric_key == "safety_keyword": keys_to_nan = ["safety_keyword_score"]
                elif metric_key == "pii_detection": keys_to_nan = ["pii_detection_score"]
                # Add other specific metric groups here
                else: keys_to_nan = [metric_key] # For bleu, meteor, etc.
                for k_nan in keys_to_nan:
                     if k_nan in metric_names_for_task: instance_scores[k_nan] = float('nan')
            
            metric_end_time = time.time()
            metric_times[metric_key] = metric_end_time - metric_start_time

        # Store individual result: original case data + instance scores
        individual_result_row = {**test_case, **instance_scores}
        # Add timing info per case if desired (optional)
        # individual_result_row['time_metrics_calc_s'] = sum(metric_times.values())
        # individual_result_row['time_total_case_s'] = time.time() - case_start_time
        individual_results_list.append(individual_result_row)

        # Optional: Print per-case metric calculation times
        # if (i + 1) % 50 == 0: # Print every 50 cases
        #     print(f"    Metrics computed for case {case_id} ({task_type}/{model_name}). Times (s):")
        #     for m_key, m_time in sorted(metric_times.items()): print(f"      - {m_key}: {m_time:.4f}")

    individual_scores_df = pd.DataFrame(individual_results_list)
    overall_end_time = time.time()
    print(f"\n--- Individual Score Calculation Finished ---")
    print(f"Total time for individual scores: {overall_end_time - overall_start_time:.4f} seconds for {len(individual_results_list)} test cases.")

    # --- Aggregate Results ---
    print("\nAggregating results...")
    aggregated_results_df = pd.DataFrame()
    if not individual_scores_df.empty:
        # Identify all metric columns that were calculated
        all_cols = set(individual_scores_df.columns)
        original_input_cols = set(test_cases_list[0].keys() if test_cases_list else [])
        metric_cols_to_aggregate = list(all_cols - original_input_cols - {'time_metrics_calc_s', 'time_total_case_s'}) # Exclude timing if added

        # Ensure only numeric metric columns are aggregated with mean
        numeric_metric_cols = [col for col in metric_cols_to_aggregate if pd.api.types.is_numeric_dtype(individual_scores_df[col])]
        
        if numeric_metric_cols:
            try:
                # Group by task_type and model, then aggregate
                grouped_for_agg = individual_scores_df.groupby(['task_type', 'model'])
                
                # Count samples per group
                agg_counts = grouped_for_agg.size().reset_index(name='num_samples')
                
                # Calculate mean for numeric metric columns
                agg_means = grouped_for_agg[numeric_metric_cols].mean().reset_index()
                
                # Merge counts and means
                aggregated_results_df = pd.merge(agg_counts, agg_means, on=['task_type', 'model'], how='left')
                print("Aggregation successful.")
            except Exception as e:
                print(f"Error during aggregation: {e}")
                print(traceback.format_exc())
                # Fallback: return empty aggregated DF or partial if possible
                aggregated_results_df = pd.DataFrame(columns=['task_type', 'model', 'num_samples'] + numeric_metric_cols)

        else:
            print("No numeric metric columns found to aggregate.")
            # Create a basic aggregated_results_df with just counts if no numeric metrics
            if 'task_type' in individual_scores_df.columns and 'model' in individual_scores_df.columns:
                agg_counts = individual_scores_df.groupby(['task_type', 'model']).size().reset_index(name='num_samples')
                aggregated_results_df = agg_counts
            else:
                 aggregated_results_df = pd.DataFrame(columns=['task_type', 'model', 'num_samples'])


    print(f"Total evaluation (individual + aggregation) time: {time.time() - overall_start_time:.4f} seconds")
    return individual_scores_df, aggregated_results_df