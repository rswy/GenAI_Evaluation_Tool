# src/metrics/conciseness.py
from .base_metric import BaseMetric
import numpy as np
import pandas as pd # For pd.isnull check

class LengthRatioMetric(BaseMetric):
    """Calculates the ratio of prediction length to reference length (word count) for a single instance."""
    def compute(self, references, predictions, **kwargs):
        # references is a single reference string
        # predictions is a single prediction string
        
        ref_str = str(references) if not pd.isnull(references) else ""
        pred_str = str(predictions) if not pd.isnull(predictions) else ""

        len_ref = len(ref_str.split())
        len_pred = len(pred_str.split())

        ratio = 1.0 # Default if both are empty
        if len_ref > 0:
            ratio = len_pred / len_ref
        elif len_pred > 0: # Reference is empty, but prediction is not
            ratio = 999.0 # Assign a large number to indicate verbosity against an empty ref
        # If both len_ref and len_pred are 0, ratio remains 1.0 (prediction is as concise as reference)
        
        # Handle potential infinity if len_ref was 0 but was not caught by above logic (should be)
        if ratio == np.inf:
            return {"length_ratio": float('nan')} # Or a large number like 999.0
            
        return {"length_ratio": ratio}