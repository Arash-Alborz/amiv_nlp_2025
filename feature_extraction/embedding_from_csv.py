# feature_extraction/embedding_from_csv.py

import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from tqdm import tqdm

# Load model once
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)
model.eval()

def get_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.squeeze().numpy()

def extract_embeddings_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = " ".join(str(row[q]) for q in ['Q1', 'Q2', 'Q3'] if pd.notna(row[q]))
        vec = get_embedding(text)
        row_dict = {f"embed_{i}": vec[i] for i in range(len(vec))}
        row_dict["id"] = row.get("id", f"row_{idx}")
        rows.append(row_dict)

    embed_df = pd.DataFrame(rows)
    return embed_df