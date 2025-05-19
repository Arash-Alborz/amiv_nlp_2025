# feature_extraction/liwc_extractor.py

import numpy as np
import re
from collections import defaultdict, Counter

# Load the LIWC dictionary
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

# getting LIWC vector from input text
def liwc_vector(text, category_map):
    tokens = re.findall(r"\b\w+\b", text.lower())
    counts = Counter()
    for category, words in category_map.items():
        for token in tokens:
            if token in words:
                counts[category] += 1
    sorted_categories = sorted(category_map.keys())
    vec = np.array([counts.get(cat, 0) for cat in sorted_categories])
    if np.sum(vec) > 0:
        vec = vec / np.sum(vec)
    return vec, sorted_categories