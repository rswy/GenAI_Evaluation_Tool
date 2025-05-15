# src/metrics/string_similarity.py
from .base_metric import BaseMetric
from .utils import safe_word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score # Retained for this file
from rouge_score import rouge_scorer # Added for consistency if this file were used
import warnings
# import statistics # No longer needed as we process single instances

class BleuMetric(BaseMetric): # NLTK version
    """Computes BLEU score for a single reference and prediction using NLTK."""
    def compute(self, references, predictions, **kwargs):
        ref_str = str(references) if references is not None else ""
        pred_str = str(predictions) if predictions is not None else ""

        ref_tokens = [safe_word_tokenize(ref_str)] # sentence_bleu expects list of reference token lists
        pred_tokens = safe_word_tokenize(pred_str)

        if not pred_tokens and not ref_tokens[0]: # Both empty
            return {"bleu": 1.0} # Or 0.0
        if not pred_tokens or not ref_tokens[0]: # One is empty
            warnings.warn(f"NLTK BLEU: Empty prediction or reference. Assigning BLEU score of 0.")
            return {"bleu": 0.0}
            
        try:
            # Using smoothing function 7 for short sentences
            score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=SmoothingFunction().method7)
        except Exception as e:
            warnings.warn(f"Could not compute NLTK BLEU for prediction: '{pred_str}'. Error: {e}. Assigning 0.")
            score = 0.0
        return {"bleu": score}

class RougeMetric(BaseMetric): # Copied from fluency_similarity for consistency
    """Computes ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) for a single reference and prediction."""
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def compute(self, references, predictions, **kwargs):
        ref_str = str(references) if references is not None else ""
        pred_str = str(predictions) if predictions is not None else ""

        if not pred_str.strip() and not ref_str.strip():
            return {"rouge_1": 1.0, "rouge_2": 1.0, "rouge_l": 1.0}
        if not pred_str.strip() or not ref_str.strip():
            warnings.warn(f"ROUGE: Empty prediction or reference. Assigning ROUGE scores of 0.")
            return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
        
        try:
            scores = self.scorer.score(ref_str, pred_str)
            rouge1_score = scores['rouge1'].fmeasure
            rouge2_score = scores['rouge2'].fmeasure
            rougeL_score = scores['rougeL'].fmeasure
        except Exception as e:
            warnings.warn(f"Could not compute ROUGE for prediction: '{pred_str}'. Error: {e}. Assigning 0.")
            rouge1_score, rouge2_score, rougeL_score = 0.0, 0.0, 0.0
        
        return {
            "rouge_1": rouge1_score,
            "rouge_2": rouge2_score,
            "rouge_l": rougeL_score
        }

class MeteorMetric(BaseMetric): # NLTK version
    """Computes METEOR score for a single tokenized reference and prediction using NLTK."""
    def compute(self, references, predictions, **kwargs):
        ref_str = str(references) if references is not None else ""
        pred_str = str(predictions) if predictions is not None else ""

        ref_tokens = safe_word_tokenize(ref_str)
        pred_tokens = safe_word_tokenize(pred_str)

        if not pred_tokens and not ref_tokens:
             return {"meteor": 1.0}
        if not pred_tokens or not ref_tokens:
            warnings.warn(f"NLTK METEOR: Empty token list for prediction or reference. Assigning METEOR score of 0.")
            return {"meteor": 0.0}
            
        try:
            # Pass token lists directly
            score = single_meteor_score(ref_tokens, pred_tokens)
        except Exception as e:
            warnings.warn(f"Could not compute NLTK METEOR for prediction tokens: '{pred_tokens}'. Error: {e}. Assigning 0.")
            score = 0.0
        
        return {"meteor": score}