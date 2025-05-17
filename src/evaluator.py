# # src/evaluator.py
# import pandas as pd
# from tasks.task_registry import (get_metrics_for_task, get_primary_reference_col,
#                                  get_primary_prediction_col, CUSTOM_METRIC_KWARG_MAP)
# from metrics import get_metric_instances
# import warnings
# from collections import defaultdict
# import traceback
# import time

# def evaluate_model_responses(test_cases_list):
#     """
#     Runs the evaluation pipeline, generating both individual and aggregated scores.
#     Assumes metric classes' compute() method now processes a single instance.
#     """
#     if not isinstance(test_cases_list, list) or not test_cases_list:
#         print("Error: Invalid or empty test_cases_list provided to evaluator.")
#         return pd.DataFrame(), pd.DataFrame()

#     individual_results_list = []
#     overall_start_time = time.time()
#     print("\nStarting Evaluation (Individual Scores)...")

#     for i, test_case in enumerate(test_cases_list):
#         case_start_time = time.time()
#         task_type = test_case.get('task_type')
#         model_name = test_case.get('model')
#         case_id = test_case.get('id', f"case_{i}")

#         if not task_type or not model_name:
#             warnings.warn(f"Skipping test case {case_id}: Missing 'task_type' or 'model'.", RuntimeWarning)
#             continue

#         # print(f"\n--- Processing Test Case ID: {case_id}, Task: {task_type}, Model: {model_name} ---") # Can be verbose

#         metric_names_for_task = get_metrics_for_task(task_type)
#         primary_ref_col = get_primary_reference_col(task_type)
#         primary_pred_col = get_primary_prediction_col(task_type)

#         if not metric_names_for_task or not primary_ref_col or not primary_pred_col:
#             warnings.warn(f"Skipping {case_id} ({task_type}/{model_name}): Missing metric names or primary column definitions.", RuntimeWarning)
#             continue

#         primary_reference = str(test_case.get(primary_ref_col, "")).strip()
#         prediction = str(test_case.get(primary_pred_col, "")).strip()

#         # Prepare kwargs for custom metrics for this single instance
#         instance_metric_kwargs = {}
#         custom_metrics_needed = {m for m in metric_names_for_task if m in CUSTOM_METRIC_KWARG_MAP}

#         if custom_metrics_needed:
#             ref_field_values_for_instance = {}
#             for metric_name_key in custom_metrics_needed: # e.g., "fact_presence_score"
#                 kwarg_map_for_metric = CUSTOM_METRIC_KWARG_MAP.get(metric_name_key, {}) # e.g., {"facts": "ref_facts"}
#                 for kwarg_name, data_column_name in kwarg_map_for_metric.items(): # e.g., kwarg_name="facts", data_column_name="ref_facts"
#                     raw_val = test_case.get(data_column_name)
#                     if pd.notna(raw_val) and str(raw_val).strip():
#                         # Ensure it's a list of strings
#                         ref_field_values_for_instance[kwarg_name] = [item.strip() for item in str(raw_val).split(',') if item.strip()]
#                     else:
#                         ref_field_values_for_instance[kwarg_name] = [] # Empty list if data missing/empty
#             instance_metric_kwargs['reference_field_values'] = ref_field_values_for_instance


#         metric_instances = get_metric_instances(metric_names_for_task)
#         instance_scores = {}
#         metric_times = {}

#         for metric_key, metric_calculator in metric_instances.items():
#             metric_start_time = time.time()
#             try:
#                 # Pass single reference, single prediction, and instance-specific kwargs
#                 computed_score_dict = metric_calculator.compute(
#                     references=primary_reference,      # Single reference string
#                     predictions=prediction,            # Single prediction string
#                     **instance_metric_kwargs           # Contains reference_field_values for this instance
#                 )
#                 if computed_score_dict:
#                     instance_scores.update(computed_score_dict)
#                 else:
#                     warnings.warn(f"Metric group '{metric_key}' compute returned None/empty for {case_id}.", RuntimeWarning)

#             except Exception as e:
#                 print(f"    *** ERROR during compute for metric group '{metric_key}' on case {case_id} ***")
#                 print(f"        Exception Type: {type(e)} | Args: {e.args}")
#                 print(f"        Traceback: {traceback.format_exc()}")
#                 # Assign NaNs for expected keys from this calculator if error
#                 keys_to_nan = []
#                 if metric_key == "rouge": keys_to_nan = ["rouge_1", "rouge_2", "rouge_l"]
#                 elif metric_key == "classification": keys_to_nan = ["accuracy", "precision", "recall", "f1_score"]
#                 elif metric_key == "fact_presence": keys_to_nan = ["fact_presence_score"]
#                 elif metric_key == "completeness": keys_to_nan = ["completeness_score"]
#                 elif metric_key == "length_ratio": keys_to_nan = ["length_ratio"]
#                 elif metric_key == "safety_keyword": keys_to_nan = ["safety_keyword_score"]
#                 elif metric_key == "pii_detection": keys_to_nan = ["pii_detection_score"]
#                 # Add other specific metric groups here
#                 else: keys_to_nan = [metric_key] # For bleu, meteor, etc.
#                 for k_nan in keys_to_nan:
#                      if k_nan in metric_names_for_task: instance_scores[k_nan] = float('nan')
            
#             metric_end_time = time.time()
#             metric_times[metric_key] = metric_end_time - metric_start_time

#         # Store individual result: original case data + instance scores
#         individual_result_row = {**test_case, **instance_scores}
#         # Add timing info per case if desired (optional)
#         # individual_result_row['time_metrics_calc_s'] = sum(metric_times.values())
#         # individual_result_row['time_total_case_s'] = time.time() - case_start_time
#         individual_results_list.append(individual_result_row)

#         # Optional: Print per-case metric calculation times
#         # if (i + 1) % 50 == 0: # Print every 50 cases
#         #     print(f"    Metrics computed for case {case_id} ({task_type}/{model_name}). Times (s):")
#         #     for m_key, m_time in sorted(metric_times.items()): print(f"      - {m_key}: {m_time:.4f}")

#     individual_scores_df = pd.DataFrame(individual_results_list)
#     overall_end_time = time.time()
#     print(f"\n--- Individual Score Calculation Finished ---")
#     print(f"Total time for individual scores: {overall_end_time - overall_start_time:.4f} seconds for {len(individual_results_list)} test cases.")

#     # --- Aggregate Results ---
#     print("\nAggregating results...")
#     aggregated_results_df = pd.DataFrame()
#     if not individual_scores_df.empty:
#         # Identify all metric columns that were calculated
#         all_cols = set(individual_scores_df.columns)
#         original_input_cols = set(test_cases_list[0].keys() if test_cases_list else [])
#         metric_cols_to_aggregate = list(all_cols - original_input_cols - {'time_metrics_calc_s', 'time_total_case_s'}) # Exclude timing if added

#         # Ensure only numeric metric columns are aggregated with mean
#         numeric_metric_cols = [col for col in metric_cols_to_aggregate if pd.api.types.is_numeric_dtype(individual_scores_df[col])]
        
#         if numeric_metric_cols:
#             try:
#                 # Group by task_type and model, then aggregate
#                 grouped_for_agg = individual_scores_df.groupby(['task_type', 'model'])
                
#                 # Count samples per group
#                 agg_counts = grouped_for_agg.size().reset_index(name='num_samples')
                
#                 # Calculate mean for numeric metric columns
#                 agg_means = grouped_for_agg[numeric_metric_cols].mean().reset_index()
                
#                 # Merge counts and means
#                 aggregated_results_df = pd.merge(agg_counts, agg_means, on=['task_type', 'model'], how='left')
#                 print("Aggregation successful.")
#             except Exception as e:
#                 print(f"Error during aggregation: {e}")
#                 print(traceback.format_exc())
#                 # Fallback: return empty aggregated DF or partial if possible
#                 aggregated_results_df = pd.DataFrame(columns=['task_type', 'model', 'num_samples'] + numeric_metric_cols)

#         else:
#             print("No numeric metric columns found to aggregate.")
#             # Create a basic aggregated_results_df with just counts if no numeric metrics
#             if 'task_type' in individual_scores_df.columns and 'model' in individual_scores_df.columns:
#                 agg_counts = individual_scores_df.groupby(['task_type', 'model']).size().reset_index(name='num_samples')
#                 aggregated_results_df = agg_counts
#             else:
#                  aggregated_results_df = pd.DataFrame(columns=['task_type', 'model', 'num_samples'])


#     print(f"Total evaluation (individual + aggregation) time: {time.time() - overall_start_time:.4f} seconds")
#     return individual_scores_df, aggregated_results_df





# src/evaluator.py
import pandas as pd
from tasks.task_registry import (get_metrics_for_task, get_primary_reference_col,
                                 get_primary_prediction_col, CUSTOM_METRIC_KWARG_MAP)
from metrics import get_metric_instances # This now includes SemanticSimilarityMetric
import warnings
from collections import defaultdict
import traceback
import time
import numpy as np # Ensure numpy is imported for float('nan')

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

    # Pre-fetch all unique metric names needed across all tasks present in test_cases
    # This allows initializing metric instances (especially heavy ones like SentenceTransformer) once.
    all_task_types_in_data = set(tc.get('task_type') for tc in test_cases_list if tc.get('task_type'))
    all_metric_names_needed = set()
    for tt in all_task_types_in_data:
        all_metric_names_needed.update(get_metrics_for_task(tt))
    
    # Initialize all metric instances once
    # This is a key optimization for models like SentenceTransformer
    # which can be slow to load repeatedly.
    print(f"Initializing metric calculators for: {all_metric_names_needed}")
    all_metric_instances = get_metric_instances(list(all_metric_names_needed))
    print("Metric calculators initialized.")


    for i, test_case in enumerate(test_cases_list):
        case_start_time = time.time()
        task_type = test_case.get('task_type')
        model_name = test_case.get('model')
        case_id = test_case.get('id', f"case_{i}")

        if not task_type or not model_name:
            warnings.warn(f"Skipping test case {case_id}: Missing 'task_type' or 'model'.", RuntimeWarning)
            continue

        metric_names_for_this_task = get_metrics_for_task(task_type)
        primary_ref_col = get_primary_reference_col(task_type)
        primary_pred_col = get_primary_prediction_col(task_type)

        if not metric_names_for_this_task or not primary_ref_col or not primary_pred_col:
            warnings.warn(f"Skipping {case_id} ({task_type}/{model_name}): Missing metric names or primary column definitions for task.", RuntimeWarning)
            continue

        primary_reference = str(test_case.get(primary_ref_col, "")).strip()
        prediction = str(test_case.get(primary_pred_col, "")).strip()

        instance_metric_kwargs = {}
        custom_metrics_needed_for_task = {m for m in metric_names_for_this_task if m in CUSTOM_METRIC_KWARG_MAP}

        if custom_metrics_needed_for_task:
            ref_field_values_for_instance = {}
            for metric_name_key in custom_metrics_needed_for_task:
                kwarg_map_for_metric = CUSTOM_METRIC_KWARG_MAP.get(metric_name_key, {})
                for kwarg_name, data_column_name in kwarg_map_for_metric.items():
                    raw_val = test_case.get(data_column_name)
                    if pd.notna(raw_val) and str(raw_val).strip():
                        ref_field_values_for_instance[kwarg_name] = [item.strip() for item in str(raw_val).split(',') if item.strip()]
                    else:
                        ref_field_values_for_instance[kwarg_name] = []
            instance_metric_kwargs['reference_field_values'] = ref_field_values_for_instance

        instance_scores = {}
        metric_times = {}

        # Use the pre-initialized metric instances relevant for this task
        for metric_key_internal, metric_calculator in all_metric_instances.items():
            # Determine which output scores this calculator is responsible for
            # This logic needs to map the internal key (e.g., "rouge") to the output keys (e.g., "rouge_1", "rouge_l")
            # that are listed in metric_names_for_this_task.
            
            # A simple check: if any of the expected output metric names for this task
            # are typically produced by this metric_calculator, then run it.
            # This mapping is implicitly handled by how get_metric_instances is structured.
            # E.g., if "rouge_1" is in metric_names_for_this_task, "rouge" instance will be fetched.
            
            # We need to ensure we only call compute if the calculator is relevant for *this task's metrics*
            # This can be a bit tricky if a calculator (like RougeMetric) produces multiple outputs (rouge_1, rouge_2, rouge_l)
            # and only some are needed for a task. The current get_metric_instances handles this by only
            # creating an instance if *any* of its output metrics are in the requested list.

            # Let's refine the check: only compute if the metric_calculator is expected to produce
            # at least one of the metric_names_for_this_task.
            # This is implicitly true due to how all_metric_instances is filtered by all_metric_names_needed.
            # However, we must ensure the `metric_key_internal` (e.g. "semantic_similarity") aligns with what `get_metric_instances` uses.

            metric_start_time = time.time()
            try:
                # The metric_calculator.compute() will return a dict of scores it calculates.
                # e.g. RougeMetric().compute() returns {'rouge_1': ..., 'rouge_2': ..., 'rouge_l': ...}
                # We only want to update instance_scores with metrics relevant to the current task.
                
                computed_score_dict = metric_calculator.compute(
                    references=primary_reference,
                    predictions=prediction,
                    **instance_metric_kwargs
                )

                if computed_score_dict:
                    for score_name, score_value in computed_score_dict.items():
                        if score_name in metric_names_for_this_task: # Only add if relevant for this task
                            instance_scores[score_name] = score_value
                else:
                    warnings.warn(f"Metric group for internal key '{metric_key_internal}' compute returned None/empty for {case_id}.", RuntimeWarning)

            except Exception as e:
                error_msg = (f"    *** ERROR during compute for metric group '{metric_key_internal}' on case {case_id} ***\n"
                             f"        Exception Type: {type(e)} | Args: {e.args}\n"
                             f"        Traceback: {traceback.format_exc()}")
                print(error_msg)
                # Attempt to assign NaNs for expected keys from this calculator if error
                # This needs to know which output keys (e.g. "rouge_1") correspond to metric_key_internal (e.g. "rouge")
                # For simplicity, we can iterate through metric_names_for_this_task and if they are typically
                # produced by this metric_key_internal and not yet in instance_scores, add NaN.
                # This part is complex to generalize perfectly without more metadata mapping internal keys to output keys.
                # A simpler approach: if a score_name from computed_score_dict (that is in metric_names_for_this_task)
                # isn't successfully computed, it just won't be in instance_scores.
                # The `streamlit_app.py` handles missing scores by showing them as NaN or "Not Applicable".
                # Let's ensure any metric in metric_names_for_this_task that *should* have been computed by this failed calculator
                # gets a NaN if it's not already set.
                
                # Example of what metric_key_internal might produce:
                potential_outputs = []
                if metric_key_internal == "bleu": potential_outputs = ["bleu"]
                elif metric_key_internal == "rouge": potential_outputs = ["rouge_1", "rouge_2", "rouge_l"]
                elif metric_key_internal == "meteor": potential_outputs = ["meteor"]
                elif metric_key_internal == "semantic_similarity": potential_outputs = ["semantic_similarity_score"]
                elif metric_key_internal == "classification": potential_outputs = ["accuracy", "precision", "recall", "f1_score"]
                elif metric_key_internal == "fact_presence": potential_outputs = ["fact_presence_score"]
                # ... and so on for all keys in METRIC_CLASS_REGISTRY used in get_metric_instances
                
                for p_out in potential_outputs:
                    if p_out in metric_names_for_this_task and p_out not in instance_scores:
                        instance_scores[p_out] = np.nan # Use np.nan
            
            metric_end_time = time.time()
            metric_times[metric_key_internal] = metric_end_time - metric_start_time

        individual_result_row = {**test_case, **instance_scores}
        individual_results_list.append(individual_result_row)

        if (i + 1) % 10 == 0 or (i + 1) == len(test_cases_list) : # Print progress
             print(f"Processed case {i+1}/{len(test_cases_list)}: ID {case_id} ({task_type}/{model_name})")


    individual_scores_df = pd.DataFrame(individual_results_list)
    overall_end_time = time.time()
    print(f"\n--- Individual Score Calculation Finished ---")
    print(f"Total time for individual scores: {overall_end_time - overall_start_time:.4f} seconds for {len(individual_results_list)} test cases.")

    # --- Aggregate Results ---
    # (Aggregation logic remains largely the same but will now include semantic_similarity_score if present)
    print("\nAggregating results...")
    aggregated_results_df = pd.DataFrame()
    if not individual_scores_df.empty:
        all_cols = set(individual_scores_df.columns)
        # Determine original input columns more dynamically if test_cases_list can be empty or diverse
        if test_cases_list:
            # Assuming all test cases have a consistent base set of keys
            # or taking keys from the first test case as representative of non-metric columns
            original_input_cols_set = set(test_cases_list[0].keys()) 
        else: # Fallback if test_cases_list was empty but somehow individual_scores_df is not
            original_input_cols_set = set(['id', 'task_type', 'model', 'question', 'ground_truth', 'answer', 
                                           'ref_facts', 'ref_key_points', 'test_description', 'contexts']) # a guess

        metric_cols_to_aggregate = list(all_cols - original_input_cols_set - {'time_metrics_calc_s', 'time_total_case_s'})
        
        # Filter for numeric metric columns before aggregation
        numeric_metric_cols = [
            col for col in metric_cols_to_aggregate 
            if pd.api.types.is_numeric_dtype(individual_scores_df[col]) and col in individual_scores_df.columns
        ]
        # Explicitly ensure 'num_samples' is not treated as a metric to average if it somehow appears here
        if 'num_samples' in numeric_metric_cols:
            numeric_metric_cols.remove('num_samples')

        if numeric_metric_cols:
            try:
                grouped_for_agg = individual_scores_df.groupby(['task_type', 'model'], dropna=False) # dropna=False for groups
                
                agg_counts = grouped_for_agg.size().reset_index(name='num_samples')
                
                # Use .mean(numeric_only=True) if pandas version supports, or stick to pre-filtered list
                agg_means = grouped_for_agg[numeric_metric_cols].mean().reset_index() 
                
                aggregated_results_df = pd.merge(agg_counts, agg_means, on=['task_type', 'model'], how='left')
                print("Aggregation successful.")
            except Exception as e:
                print(f"Error during aggregation: {e}")
                print(traceback.format_exc())
                aggregated_results_df = pd.DataFrame(columns=['task_type', 'model', 'num_samples'] + numeric_metric_cols)
        else:
            print("No numeric metric columns found to aggregate for means.")
            if 'task_type' in individual_scores_df.columns and 'model' in individual_scores_df.columns:
                agg_counts = individual_scores_df.groupby(['task_type', 'model'], dropna=False).size().reset_index(name='num_samples')
                aggregated_results_df = agg_counts
            else:
                 aggregated_results_df = pd.DataFrame(columns=['task_type', 'model', 'num_samples'])

    print(f"Total evaluation (individual + aggregation) time: {time.time() - overall_start_time:.4f} seconds")
    return individual_scores_df, aggregated_results_df
# ```
# **Note on `evaluator.py`:** The main change here is an *optimization*: initializing metric instances (like `SentenceTransformer`) once at the beginning rather than per task or per test case. This significantly speeds up evaluations when using heavy models. The rest of the logic should adapt to the new semantic similarity metric as it flows through `get_metric_instance