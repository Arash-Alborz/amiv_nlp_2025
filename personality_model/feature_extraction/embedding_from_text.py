# feature_extraction/embedding_extractor.py

from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np

# Load on CPU
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)
model.eval()

def get_bert_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (1, 768)
        return cls_embedding.squeeze().numpy()