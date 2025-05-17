# src/metrics/fluency_similarity.py
from .base_metric import BaseMetric
from .utils import safe_word_tokenize # Assuming this utility is still relevant for other metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
import warnings
import numpy as np
import os # For checking local model path

# Attempt to import sentence-transformers and PyTorch utilities
SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    warnings.warn(
        "Sentence Transformers library not found. SemanticSimilarityMetric will not be available. "
        "Please install it with `pip install sentence-transformers`.",
        ImportWarning
    )

# Define a default model - 'all-MiniLM-L6-v2' is a good balance of speed and performance.
DEFAULT_ST_MODEL = 'all-MiniLM-L6-v2'
# Define a potential local path (users should configure this or ensure model is in this relative path)
DEFAULT_LOCAL_MODEL_PATH_SEMANTIC = "./models/all-MiniLM-L6-v2" 

class BleuMetric(BaseMetric):
    """Computes BLEU score for a single reference and prediction using NLTK."""
    def compute(self, references, predictions, **kwargs):
        ref_str = str(references) if references is not None else ""
        pred_str = str(predictions) if predictions is not None else ""

        ref_tokens = [safe_word_tokenize(ref_str)] 
        pred_tokens = safe_word_tokenize(pred_str)

        if not pred_tokens and not ref_tokens[0]:
            return {"bleu": 1.0}
        if not pred_tokens or not ref_tokens[0]:
            warnings.warn("NLTK BLEU: Empty prediction or reference. Assigning BLEU score of 0.")
            return {"bleu": 0.0}
            
        try:
            score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=SmoothingFunction().method7)
        except Exception as e:
            warnings.warn(f"Could not compute NLTK BLEU for prediction: '{pred_str}'. Error: {e}. Assigning 0.")
            score = 0.0
        return {"bleu": score}

class RougeMetric(BaseMetric):
    """Computes ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) for a single reference and prediction."""
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def compute(self, references, predictions, **kwargs):
        ref_str = str(references) if references is not None else ""
        pred_str = str(predictions) if predictions is not None else ""

        if not pred_str.strip() and not ref_str.strip():
            return {"rouge_1": 1.0, "rouge_2": 1.0, "rouge_l": 1.0}
        if not pred_str.strip() or not ref_str.strip():
            warnings.warn("ROUGE: Empty prediction or reference. Assigning ROUGE scores of 0.")
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

class MeteorMetric(BaseMetric):
    """Computes METEOR score for a single tokenized reference and prediction using NLTK."""
    def compute(self, references, predictions, **kwargs):
        ref_str = str(references) if references is not None else ""
        pred_str = str(predictions) if predictions is not None else ""

        ref_tokens = safe_word_tokenize(ref_str)
        pred_tokens = safe_word_tokenize(pred_str)

        if not pred_tokens and not ref_tokens:
             return {"meteor": 1.0}
        if not pred_tokens or not ref_tokens:
            warnings.warn("NLTK METEOR: Empty token list for prediction or reference. Assigning METEOR score of 0.")
            return {"meteor": 0.0}
            
        try:
            score = single_meteor_score(ref_tokens, pred_tokens) # NLTK's single_meteor_score expects tokenized strings
        except Exception as e:
            warnings.warn(f"Could not compute NLTK METEOR for prediction tokens: '{pred_tokens}'. Error: {e}. Assigning 0.")
            score = 0.0
        
        return {"meteor": score}

class SemanticSimilarityMetric(BaseMetric):
    """
    Computes semantic similarity between a reference and a prediction string
    using sentence embeddings from the Sentence Transformers library.
    Attempts to load model from a local path first, then from Hugging Face Hub.
    """
    def __init__(self, model_name_or_path=DEFAULT_ST_MODEL, local_model_path=DEFAULT_LOCAL_MODEL_PATH_SEMANTIC):
        
        """
        Initializes the SemanticSimilarityMetric.

        Args:
            model_name_or_path (str): The name of the pre-trained sentence-transformer model (e.g., 'all-MiniLM-L6-v2')
                                     or path to a local model. This is used if local_model_path is not found.
            local_model_path (str): Preferred path to a locally downloaded model.
        """
        self.model = None
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            warnings.warn("Sentence Transformers library not available. Semantic similarity will return NaN.", RuntimeWarning)
            return
        
        # Try loading from the specified local_model_path first

        if local_model_path and os.path.exists(local_model_path) and os.path.isdir(local_model_path):
            try:
                self.model = SentenceTransformer(local_model_path)
                print(f"SemanticSimilarityMetric: Loaded model from local path: {local_model_path}")
            except Exception as e:
                warnings.warn(f"Failed to load Sentence Transformer model from local path '{local_model_path}': {e}. "
                              f"Attempting to load '{model_name_or_path}' from Hub.", RuntimeWarning)
                self.model = None  # Ensure model is None before trying hub
        
        # If local loading failed or path not provided/valid, try loading from model_name_or_path (could be Hub name)
        if self.model is None: # This will only be attempted if local loading failed AND internet is available
            try:
                self.model = SentenceTransformer(model_name_or_path) 
                # This line would attempt an internet download if the model_name_or_path is not a local path
                # and not found in cache. In a truly offline scenario, this should ideally not be reached
                # if the local path was correctly set up and the model exists there.
                print(f"SemanticSimilarityMetric: Loaded model '{model_name_or_path}' (likely from Hub or cache).")
            except Exception as e:
                warnings.warn(f"Failed to load Sentence Transformer model '{model_name_or_path}': {e}. "
                              "Semantic similarity will return NaN.", RuntimeWarning)
                self.model = None

    def compute(self, references, predictions, **kwargs):
        """
        Computes the cosine similarity between the embeddings of the reference and prediction.
        """
        if self.model is None: # If model failed to load in __init__
            return {"semantic_similarity_score": np.nan}

        ref_str = str(references) if references is not None else ""
        pred_str = str(predictions) if predictions is not None else ""

        if not ref_str.strip() and not pred_str.strip():
            return {"semantic_similarity_score": 1.0} 
        if not ref_str.strip() or not pred_str.strip():
            return {"semantic_similarity_score": 0.0} 

        try:
            embedding_reference = self.model.encode(ref_str, convert_to_tensor=True)
            embedding_prediction = self.model.encode(pred_str, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(embedding_reference, embedding_prediction)
            similarity_score = cosine_scores.item()
            return {"semantic_similarity_score": similarity_score}
        except Exception as e:
            warnings.warn(f"Error computing semantic similarity for instance. "
                          f"Reference: '{ref_str[:50]}...', Prediction: '{pred_str[:50]}...'. Error: {e}",
                          RuntimeWarning)
            return {"semantic_similarity_score": np.nan}

