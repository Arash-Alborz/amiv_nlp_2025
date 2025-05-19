# personality_model.py

import numpy as np
import joblib
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import re
from collections import Counter, defaultdict

TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Emotional stability"]
LABELS = ["low", "medium", "high"]

def load_liwc_dic(dic_path):
    category_map = defaultdict(list)
    with open(dic_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' not in line:
                continue
            parts = line.strip().split()
            category = parts[0].rstrip(':')
            words = parts[1:]
            category_map[category] = words
    return category_map

def liwc_embedding(text, category_map, sorted_categories):
    tokens = re.findall(r"\b\w+\b", text.lower())
    counts = Counter()
    for category, words in category_map.items():
        for token in tokens:
            if token in words:
                counts[category] += 1
    vec = np.array([counts.get(cat, 0) for cat in sorted_categories])
    return vec

def get_embedding(text, tokenizer, model):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cpu")
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

class PersonalityClassifier:
    def __init__(self):
        # DistilBERT embedding extraction
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert/distilbert-base-cased-distilled-squad")
        self.bert_model = DistilBertModel.from_pretrained("distilbert/distilbert-base-cased-distilled-squad")
        self.bert_model.eval()

        # scaler for scaling embeddings + liwc
        self.scaler = joblib.load("models/feature_scaler.pkl")

        # classifiers
        self.models = {}
        for trait in TRAITS:
            path = f"models/{trait.lower().replace(' ', '_')}_classifier.pkl"
            self.models[trait] = joblib.load(path)

        # LIWC
        self.liwc_map = load_liwc_dic("models/output.dic")
        self.sorted_liwc_cats = sorted(self.liwc_map.keys())

    def extract_features(self, text):
        embed = get_embedding(text, self.tokenizer, self.bert_model)
        liwc = liwc_embedding(text, self.liwc_map, self.sorted_liwc_cats)
        if np.sum(liwc) > 0:
            liwc = liwc / np.sum(liwc)
        combined = np.concatenate([embed, liwc])
        return combined.reshape(1, -1)

    def predict_all_traits(self, text):
        features = self.extract_features(text)
        scaled = self.scaler.transform(features)

        results = {}
        for trait in TRAITS:
            clf = self.models[trait]
            pred = clf.predict(scaled)[0]
            results[trait] = pred  #  string: "low", "medium", or "high"
        return results