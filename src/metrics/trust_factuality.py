# src/metrics/trust_factuality.py
from .base_metric import BaseMetric
import warnings
# import re # Not used here
import numpy as np # For float('nan')
import pandas as pd # For pd.notna, though str checks often suffice

class FactPresenceMetric(BaseMetric):
    """
    Checks for the presence of predefined facts in a single prediction.
    Assumes facts are provided as a list in reference_field_values for the instance.
    Returns NaN if no facts are provided for checking.
    """
    def compute(self, references, predictions, **kwargs):
        # `references` is the primary reference text for the single instance (not directly used by this metric)
        # `predictions` is the single prediction string for the instance
        # `kwargs` contains `reference_field_values` for the single instance
        # e.g., {'facts': ['fact A for this case', 'fact B for this case']}
        
        prediction_str = str(predictions) if predictions is not None else ""
        ref_values_for_instance = kwargs.get('reference_field_values', {})
        
        facts_list_for_instance = ref_values_for_instance.get('facts', [])

        if not isinstance(facts_list_for_instance, list) or not facts_list_for_instance:
            # If no facts are provided to check against, the metric is not applicable.
            return {"fact_presence_score": np.nan} 

        pred_lower = prediction_str.lower()
        facts_found = 0
        for fact in facts_list_for_instance:
            if fact: # Ensure fact is not None or empty
                fact_lower = str(fact).lower().strip()
                if fact_lower in pred_lower:
                    facts_found += 1
        
        # Score is calculated only if facts_list_for_instance is not empty (handled by the NaN return above)
        instance_score = facts_found / len(facts_list_for_instance)
        
        return {"fact_presence_score": instance_score}