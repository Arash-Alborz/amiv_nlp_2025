# feature_extraction/pipeline.py

import numpy as np
import joblib

from feature_extraction.embedding_from_text import get_bert_embedding
from feature_extraction.liwc_from_text import load_liwc_dic, liwc_vector

# Load the LIWC lexicon once
liwc_map = load_liwc_dic("models/output.dic")

# Load the scaler
scaler = joblib.load("models/scaler.pkl")

def text_to_features(text: str) -> np.ndarray:
    # Get BERT embedding (768-dim)
    emb_vec = get_bert_embedding(text)

    # Get LIWC vector (~64-dim)
    liwc_vec, _ = liwc_vector(text, liwc_map)

    # Combine into one long vector
    full_vec = np.concatenate([emb_vec, liwc_vec])

    # Standardize using the saved scaler
    scaled_vec = scaler.transform([full_vec])  # shape: (1, total_dim)

    return scaled_vec  # Return the standardized vector for prediction