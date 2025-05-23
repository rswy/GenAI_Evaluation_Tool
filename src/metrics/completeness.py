# src/metrics/completeness.py
from .base_metric import BaseMetric
import warnings
import numpy as np # For float('nan')
import pandas as pd # For pd.notna if used, though str conversion handles most

class ChecklistCompletenessMetric(BaseMetric):
    """
    Checks for the presence of predefined key points/checklist items in a single prediction.
    Assumes key points are provided as a list in reference_field_values for the instance.
    Returns NaN if no key points are provided for checking.
    """
    def compute(self, references, predictions, **kwargs):
        # `references` is the primary reference text for the single instance (might not be used by this metric)
        # `predictions` is the single prediction string for the instance
        # `kwargs` contains `reference_field_values` for the single instance
        # e.g., {'key_points': ['point A for this case', 'point B for this case']}
        
        prediction_str = str(predictions) if predictions is not None else ""
        ref_values_for_instance = kwargs.get('reference_field_values', {})
        
        key_points_list_for_instance = ref_values_for_instance.get('key_points', [])

        if not isinstance(key_points_list_for_instance, list) or not key_points_list_for_instance:
            # If no key points are provided to check against, the metric is not applicable.
            return {"completeness_score": np.nan}

        pred_lower = prediction_str.lower()
        points_found = 0
        for point in key_points_list_for_instance:
            if point: # Ensure point is not None or empty
                point_lower = str(point).lower().strip()
                if point_lower in pred_lower:
                    points_found += 1
        
        # Score is calculated only if key_points_list_for_instance is not empty (handled by the NaN return above)
        instance_score = points_found / len(key_points_list_for_instance)
        
        return {"completeness_score": instance_score}