# classification_model.py
# GPU-enabled with safe CPU fallback and Windows stability fixes

import os
import warnings

# Critical Windows stability fixes - MUST be set before heavy imports
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

class DetoxifyWrapper:
    def __init__(self, model_type: str = "original", device: str = "cpu"):
        self.model_type = model_type
        self.device = device  # <— do NOT force CPU here

        try:
            print(f"Loading Detoxify '{model_type}' on {self.device}...")
            self.detox = Detoxify(model_type=self.model_type, device=self.device)
            print("Model loaded successfully")
        except Exception as e:
            # If GPU load fails, transparently fall back to CPU + original
            print(f"Error loading model on {self.device}: {e}")
            try:
                if self.device != "cpu":
                    print("Falling back to CPU…")
                self.detox = Detoxify(model_type='original', device='cpu')
                self.model_type = 'original'
                self.device = 'cpu'
                print("Fallback to CPU 'original' model successful")
            except Exception as e2:
                print(f"All model load attempts failed: {e2}")
                raise

    def predict(self, texts, return_probability: bool = True):
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        texts = [str(text).strip() if text else "empty" for text in texts]

        try:
            with torch.no_grad():
                scores = self.detox.predict(texts)

            results = []
            toxicity_scores = scores.get("toxicity", [])

            for i, score in enumerate(toxicity_scores):
                toxicity = float(score)
                toxicity = max(0.0, min(1.0, toxicity))

                severe_tox = float(scores.get("severe_toxicity", [0.0] * len(toxicity_scores))[i]) if "severe_toxicity" in scores else 0.0
                obscene = float(scores.get("obscene", [0.0] * len(toxicity_scores))[i]) if "obscene" in scores else 0.0
                threat = float(scores.get("threat", [0.0] * len(toxicity_scores))[i]) if "threat" in scores else 0.0
                insult = float(scores.get("insult", [0.0] * len(toxicity_scores))[i]) if "insult" in scores else 0.0
                identity = float(scores.get("identity_attack", [0.0] * len(toxicity_scores))[i]) if "identity_attack" in scores else 0.0

                label = "hateful" if toxicity >= 0.5 else "non-hateful"

                if return_probability:
                    results.append({
                        "label": label,
                        "probability": {"hateful": toxicity, "non-hateful": 1.0 - toxicity},
                        "toxicity": toxicity,
                        "severe_toxicity": max(0.0, min(1.0, severe_tox)),
                        "obscene": max(0.0, min(1.0, obscene)),
                        "threat": max(0.0, min(1.0, threat)),
                        "insult": max(0.0, min(1.0, insult)),
                        "identity_attack": max(0.0, min(1.0, identity))
                    })
                else:
                    results.append(label)

            return results[0] if single_input else results

        except Exception as e:
            print(f"Prediction error: {e}")
            fallback = {
                "label": "non-hateful", "probability": {"hateful": 0.0, "non-hateful": 1.0},
                "toxicity": 0.0, "severe_toxicity": 0.0, "obscene": 0.0,
                "threat": 0.0, "insult": 0.0, "identity_attack": 0.0
            } if return_probability else "non-hateful"

            return fallback if single_input else [fallback] * len(texts)


class ModelLoader:
    def __init__(self, device: str | None = None):
        # Prefer explicit device; otherwise auto-select
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Optional env override: DETOXIFY_MODEL_TYPE=original|multilingual
        model_type = os.environ.get("DETOXIFY_MODEL_TYPE", "original")

        print("Initializing ModelLoader")
        print(f"Requested device: {device.upper()}")
        if device == "cuda":
            try:
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            except Exception:
                pass
            torch.backends.cudnn.benchmark = True
            # allow TF32 where possible for speed on Ampere+
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
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

    def device_check(self):
        if self.device == "cuda":
            print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
            print(f"PyTorch: {torch.__version__}")
        else:
            print("Using device: CPU")
            print(f"PyTorch: {torch.__version__}")

    def load_model(self):
        self.device_check()
        return self.model, None


class ModelEvaluation:
    def __init__(self, texts, tokenizer=None):
        self.texts = texts
        self.padding = "max_length"
        self.truncation = True
        self.max_length = 128
        self.tokenizer = tokenizer
