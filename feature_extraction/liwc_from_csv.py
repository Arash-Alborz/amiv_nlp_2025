# feature_extraction/liwc_from_csv.py

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import re

def load_liwc_dic(dic_path="models/output.dic"):
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

def extract_liwc_from_csv(csv_path, category_map):
    df = pd.read_csv(csv_path)
    sorted_categories = sorted(category_map.keys())

    def process_row(row):
        text = " ".join(str(row[q]) for q in ['Q1', 'Q2', 'Q3'] if pd.notna(row[q]))
        tokens = re.findall(r"\b\w+\b", text.lower())
        counts = Counter()
        for category, words in category_map.items():
            for token in tokens:
                if token in words:
                    counts[category] += 1
        vec = np.array([counts.get(cat, 0) for cat in sorted_categories])
        if np.sum(vec) > 0:
            vec = vec / np.sum(vec)
        return vec

    liwc_features = df.apply(process_row, axis=1, result_type="expand")
    liwc_features.columns = [f"liwc_{cat}" for cat in sorted_categories]
    return liwc_features