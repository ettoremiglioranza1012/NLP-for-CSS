# classification_model.py
# This script loads a TweetNLP model for hate detection and evaluates it
import tweetnlp
import torch


class ModelLoader:
    def __init__(self):
        self.model_name = 'twitter-roberta-base-hate-latest'
        self.model = tweetnlp.Classifier("cardiffnlp/twitter-roberta-base-hate-latest")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def device_check(self):
        print(f"Using device: {self.device}")

    def load_model(self):
        self.device_check()
        return self.model, None  # TweetNLP handles tokenization internally


class ModelEvaluation:
    def __init__(self, texts, tokenizer=None):
        self.texts = texts
        self.padding = 'max_length'
        self.truncation = True
        self.max_length = 128
        self.tokenizer = tokenizer  # Not needed for TweetNLP but kept for compatibility


def main():
    # Initialize the model loader
    mod = ModelLoader()
    # Check the device and load the model
    mod.device_check()
    # Load the model and tokenizer
    model, tokenizer = mod.load_model()
    # Model name check
    print(f"Model loaded: {mod.model_name}")


if __name__ == "__main__":
    main()