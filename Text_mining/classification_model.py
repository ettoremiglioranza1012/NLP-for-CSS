# classification_model.py
# GPU-enabled with safe CPU fallback and Windows stability fixes

import os
import warnings
from typing import Any, Dict, List, Optional, Union

# --- Critical Windows stability fixes: MUST be set before heavy imports ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")

import torch
from detoxify import Detoxify

# Keep single-threaded CPU math when we do fall back to CPU
torch.set_num_threads(1)
torch.set_grad_enabled(False)

VALID_MODELS = {"original", "unbiased", "multilingual"}


class DetoxifyWrapper:
    """
    Thin convenience wrapper around Detoxify to normalize outputs and provide a safe
    GPU->CPU fallback. Supports 'original', 'unbiased', and 'multilingual'.
    """

    def __init__(self, model_type: str = "original", device: str = "cpu"):
        if model_type not in VALID_MODELS:
            print(f"Unknown model_type '{model_type}', falling back to 'unbiased'")
            model_type = "unbiased"

        self.model_type = model_type
        self.device = device  # do NOT force CPU here

        try:
            print(f"Loading Detoxify '{self.model_type}' on {self.device}...")
            self.detox = Detoxify(model_type=self.model_type, device=self.device)
            print("Model loaded successfully")
        except Exception as e:
            # If GPU load fails, transparently fall back to CPU + original
            print(f"Error loading model on {self.device}: {e}")
            try:
                if self.device != "cpu":
                    print("Falling back to CPU…")
                self.detox = Detoxify(model_type="original", device="cpu")
                self.model_type = "original"
                self.device = "cpu"
                print("Fallback to CPU 'original' model successful")
            except Exception as e2:
                print(f"All model load attempts failed: {e2}")
                raise

    def predict(
        self,
        texts: Union[str, List[str]],
        return_probability: bool = True,
        threshold: float = 0.5,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]], str, List[str]]:
        """
        Run predictions. If return_probability=True, returns rich dict(s) with scores.
        Otherwise returns label(s) 'hateful'/'non-hateful' based on `threshold`.
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Normalize inputs; replace None/empty with a neutral token
        norm_texts: List[str] = [str(t).strip() if t else "empty" for t in texts]

        try:
            with torch.no_grad():
                scores: Dict[str, List[float]] = self.detox.predict(norm_texts)

            # The Detoxify API returns a dict of lists keyed by category names
            toxicity_scores = scores.get("toxicity", [0.0] * len(norm_texts))
            n = len(toxicity_scores)

            def get_key(k: str) -> List[float]:
                vals = scores.get(k)
                if vals is None or len(vals) != n:
                    return [0.0] * n
                return vals

            severe_vals = get_key("severe_toxicity")
            obscene_vals = get_key("obscene")
            threat_vals = get_key("threat")
            insult_vals = get_key("insult")
            identity_vals = get_key("identity_attack")

            results: List[Union[str, Dict[str, Any]]] = []
            for i in range(n):
                toxicity = float(toxicity_scores[i])
                # Clamp to [0,1]
                toxicity = max(0.0, min(1.0, toxicity))

                severe_tox = max(0.0, min(1.0, float(severe_vals[i])))
                obscene = max(0.0, min(1.0, float(obscene_vals[i])))
                threat = max(0.0, min(1.0, float(threat_vals[i])))
                insult = max(0.0, min(1.0, float(insult_vals[i])))
                identity = max(0.0, min(1.0, float(identity_vals[i])))

                label = "hateful" if toxicity >= threshold else "non-hateful"

                if return_probability:
                    results.append(
                        {
                            "label": label,
                            "probability": {
                                "hateful": toxicity,
                                "non_hateful": 1.0 - toxicity,
                            },
                            "toxicity": toxicity,
                            "severe_toxicity": severe_tox,
                            "obscene": obscene,
                            "threat": threat,
                            "insult": insult,
                            "identity_attack": identity,
                            "model_type": self.model_type,
                            "device": self.device,
                        }
                    )
                else:
                    results.append(label)

            return results[0] if single_input else results

        except Exception as e:
            print(f"Prediction error: {e}")
            # Safe neutral fallback
            fallback_rich = {
                "label": "non-hateful",
                "probability": {"hateful": 0.0, "non_hateful": 1.0},
                "toxicity": 0.0,
                "severe_toxicity": 0.0,
                "obscene": 0.0,
                "threat": 0.0,
                "insult": 0.0,
                "identity_attack": 0.0,
                "model_type": self.model_type,
                "device": self.device,
            }
            if single_input:
                return fallback_rich if return_probability else "non-hateful"
            return [fallback_rich if return_probability else "non-hateful"] * len(norm_texts)


class ModelLoader:
    """
    Selects device, validates model type (via env DETOXIFY_MODEL_TYPE) and
    initializes DetoxifyWrapper with safe CUDA tweaks.
    """

    def __init__(self, device: Optional[str] = None):
        # Prefer explicit device; otherwise auto-select
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Optional env override: DETOXIFY_MODEL_TYPE=original|unbiased|multilingual
        model_type = os.environ.get("DETOXIFY_MODEL_TYPE", "unbiased")
        if model_type not in VALID_MODELS:
            print(f"Unknown DETOXIFY_MODEL_TYPE='{model_type}', falling back to 'unbiased'")
            model_type = "unbiased"

        print("Initializing ModelLoader")
        print(f"Requested device: {device.upper()}")

        if device == "cuda":
            try:
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            except Exception:
                pass
            # cuDNN autotune for fixed shapes
            torch.backends.cudnn.benchmark = True
            # Allow TF32 for speed on Ampere+
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                # For PyTorch 2.x this is a no-op on older GPUs
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        try:
            self.model = DetoxifyWrapper(model_type=model_type, device=device)
            self.model_name = f"detoxify-{self.model.model_type}"
            self.device = device
        except Exception as e:
            print(f"Failed to initialize model: {e}")
            raise

    def device_check(self) -> None:
        if self.device == "cuda":
            try:
                dev_name = torch.cuda.get_device_name(0)
            except Exception:
                dev_name = "Unknown CUDA device"
            print(f"Using device: CUDA ({dev_name})")
            print(f"PyTorch: {torch.__version__}")
        else:
            print("Using device: CPU")
            print(f"PyTorch: {torch.__version__}")

    def load_model(self):
        self.device_check()
        # second value kept for backward compatibility with previous callers
        return self.model, None


class ModelEvaluation:
    """
    Kept for backward compatibility with previous pipelines. Tokenizer is unused
    because Detoxify handles its own preprocessing.
    """

    def __init__(self, texts: List[str], tokenizer: Any = None):
        self.texts = texts
        self.padding = "max_length"
        self.truncation = True
        self.max_length = 128
        self.tokenizer = tokenizer
