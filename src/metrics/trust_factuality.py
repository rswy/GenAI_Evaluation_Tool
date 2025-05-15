# src/metrics/trust_factuality.py
from .base_metric import BaseMetric
import warnings
# import re # Not used here
# import numpy as np # Not used
import pandas as pd # For pd.notna, though str checks often suffice

class FactPresenceMetric(BaseMetric):
    """
    Checks for the presence of predefined facts in a single prediction.
    Assumes facts are provided as a list in reference_field_values for the instance.
    """
    def compute(self, references, predictions, **kwargs):
        # `references` is the primary reference text for the single instance (not directly used by this metric)
        # `predictions` is the single prediction string for the instance
        # `kwargs` contains `reference_field_values` for the single instance
        # e.g., {'facts': ['fact A for this case', 'fact B for this case']}
        
        prediction_str = str(predictions) if predictions is not None else ""
        ref_values_for_instance = kwargs.get('reference_field_values', {})
        
        # Debugging print
        # print(f"\n--- DEBUG FactPresence Instance ---")
        # print(f"  Prediction (lower): '{prediction_str.lower()[:200]}...'")
        # print(f"  Ref Values for Instance: {ref_values_for_instance}")

        facts_list_for_instance = ref_values_for_instance.get('facts', [])

        if not isinstance(facts_list_for_instance, list) or not facts_list_for_instance:
            # print(f"  No facts list or empty. Score: 0.0")
            # print(f"--- End DEBUG FactPresence Instance ---\n")
            return {"fact_presence_score": 0.0} # Or float('nan')

        pred_lower = prediction_str.lower()
        facts_found = 0
        for fact in facts_list_for_instance:
            if fact: # Ensure fact is not None or empty
                fact_lower = str(fact).lower().strip()
                if fact_lower in pred_lower:
                    facts_found += 1
        
        instance_score = facts_found / len(facts_list_for_instance) if facts_list_for_instance else 0.0
        
        # print(f"  Facts Found: {facts_found} / {len(facts_list_for_instance)}")
        # print(f"  Instance Score: {instance_score}")
        # print(f"--- End DEBUG FactPresence Instance ---\n")
        return {"fact_presence_score": instance_score}