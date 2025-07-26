
# data_classification.py
# This script loads a fine-tuned RoBERTa model for sequence classification and evaluates it
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch


class ModelLoader:
    def __init__(self, path="./RobWgs"):
        self.model_name = 'FreeGazaRoberta'
        self.model_path = path
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(self.model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def device_check(self):
        print(f"Using device: {self.device}")
    
    def load_model(self):
        self.device_check()
        return self.model, self.tokenizer


class ModelEvaluation:
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.padding = 'max_length'
        self.truncation = True
        self.max_length = 128
        self.tokenizer = tokenizer 


def main():

    # Initialize the model loader
    mod = ModelLoader()
    # Check the device and load the model
    mod.device_check()
    # Load the model and tokenizer
    model, tokenizer = mod.load_model()

    # Model name check
    print(f"Model loaded:{mod.model_name}")
    

