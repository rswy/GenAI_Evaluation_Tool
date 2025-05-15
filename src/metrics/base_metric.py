# src/metrics/base_metric.py
from abc import ABC, abstractmethod

class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics."""
    @abstractmethod
    def compute(self, references, predictions, **kwargs):
        """
        Computes the metric score(s) for a single instance.

        Args:
            references (object): Ground truth reference for a single instance.
                                 Can be a string, label, or other structure.
            predictions (object): Model prediction for a single instance.
                                  Format should generally match references.
            **kwargs: Additional arguments specific to the metric for the single instance,
                      e.g., reference_field_values={'facts': [...], 'key_points': [...]}
                      for the current instance.

        Returns:
            dict: A dictionary where keys are metric names (e.g., 'rouge_l', 'accuracy')
                  and values are the computed scores for this single instance.
        """
        pass