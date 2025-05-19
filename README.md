# Personality Trait Predictor — AMIV NLP 2025
University of Antwerp

This project predicts **Big Five personality traits (OCEAN)** from English text using a combination of:

- DistilBERT embeddings  
- LIWC-style psycholinguistic features  
- 5 Custom-trained Random Forest classifiers

Each of the five traits is predicted as a categorical label:
- `low`  
- `medium`  
- `high`

### Traits Predicted
- **Openness**
- **Conscientiousness**
- **Extraversion**
- **Agreeableness**
- **Emotional Stability**

---

## Project Overview

**Important**: hf_personality_model/ folder is for itself a nested-git that is hosted on HuggingFace. The content of this folder cannot be seen on GitHub.
The trained personality model (hf_personality_model/) is hosted separately on Hugging Face and not included in this repository. You can find it here: https://huggingface.co/Arash-Alborz/personality-trait-predictor”

This repository contains all components of the project:

- **preprocessingn/**: Code for extracting BERT embeddings and LIWC-style features / Scripts for binning continuous scores, cleaning, and formatting raw data
- **classification/**: Training, evaluation, and grid search scripts for different classifiers
- **synthetic_data_generation/**: Scripts for generating synthetic job interview-style responses using OpenAI GPT
- **data_analysis/**: Tools for comparing classifiers, saving performance reports, and generating visual summaries
- **processed_data/**: liwc features and embeddings are available as csv files for training and further optimization
- **hf_personality_model/**: A separate folder containing the deployable Hugging Face model (trained)

---

## Quick Usage Example (Prediction) on HuggingFace

```python
from personality_model import PersonalityClassifier

model = PersonalityClassifier()

text = "I love exploring new cultures and trying unusual foods. I often seek out unfamiliar ideas and perspectives."
result = model.predict_all_traits(text)
print(result)
```

Expected output:
```python
{
  "Openness": "high",
  "Conscientiousness": "medium",
  "Extraversion": "low",
  "Agreeableness": "high",
  "Emotional stability": "medium"
}
```

---

## Project Structure

```
amiv_nlp_2025/
├── personality_model.py              # Prediction pipeline
├── test_personality_model.py         # Script for testing raw text input
├── predict_from_csv.ipynb            # Apply model to CSV files with Q1/Q2/Q3 responses
├── requirements.txt
├── README.md
│
├── models/                           # Pretrained classifier models + scaler + LIWC dictionary
│   ├── openness_classifier.pkl
│   ├── conscientiousness_classifier.pkl
│   ├── extraversion_classifier.pkl
│   ├── agreeableness_classifier.pkl
│   ├── emotional_stability_classifier.pkl
│   ├── feature_scaler.pkl
│   └── output.dic
│
├── feature_extraction/               # Feature engineering
│   ├── embedding_from_text.py
│   ├── liwc_from_text.py
│   └── __init__.py
│
├── classification/                   # Classifier training and evaluation
│   ├── rf_classifier.py
│   ├── mlp_classifier.py
│   ├── gridsearch_rf.py
│   ├── ...
│
├── preprocessing/                    # Cleaning, formatting, and binning logic
│   ├── bin_scores.py
│   ├── clean_data.py
│   ├── ...
│
├── synthetic_data/                   # GPT-based synthetic response generator
│   ├── generate_synthetic_data.py
│   ├── synthetic_token_tracking.csv
│   └── synthetic_val_data.csv
│
├── evaluation/                       # Evaluation and comparison tools
│   ├── model_comparison.csv
│   ├── results_visualization.py
│   └── ...
│
├── hf_personality_model/            # [Deployed Hugging Face model folder]
│   ├── personality_model.py
│   ├── models/ (copied subset)
│   ├── predict_from_csv.ipynb
│   ├── README.md
│   └── ...
```

---

## Modeling Details

- **Embedding Model**: `distilbert-base-cased-distilled-squad`
- **LIWC Dictionary**: `output.dic` (64 dimensions)
- **Classifier Types**:
  - `RandomForestClassifier`
  - `GradientBoostingClassifier`
  - `MLPClassifier`
  - `SVC`
  - `Logistic Regression`
- Each trait has a separate classifier
- Features are scaled with `StandardScaler`
- Continuous values are converted to labels using the following rules:

| Score Range       | Label   |
|-------------------|---------|
| 0 ≤ score ≤ 32    | Low     |
| 33 ≤ score ≤ 66   | Medium  |
| 67 ≤ score ≤ 100  | High    |

---

## Evaluation Summary

The final optimized model (class PersonalityClassifier) consists of 5 random forest classifiers each trained separately for a trait, achieving the final scores as follows:

| Trait               | Accuracy | Macro F1-score |
|---------------------|----------|----------------|
| Openness            | 0.62     | 0.47           |
| Conscientiousness   | 0.62     | 0.48           |
| Extraversion        | 0.47     | 0.44           |
| Agreeableness       | 0.38     | 0.36           |
| Emotional stability | 0.53     | 0.47           |

All the classification reports can be found in .txt format in classification/reports
All final models with fixed weights are stored in `models/`. -> HuggingFace

---

## Installation

We recommend creating a conda environment.

```bash
conda create -n amiv_nlp_2025 python=3.9
conda activate amiv_nlp_2025
pip install -r requirements.txt
```

---

## Hugging Face Model

A clean version of the model (ready for deployment) is stored in `hf_personality_model/`. This version contains:
- All necessary pretrained `.pkl` files
- `personality_model.py`
- `predict_from_csv.ipynb`

Model hosted at: [https://huggingface.co/Arash-Alborz/personality-trait-predictor](https://huggingface.co/Arash-Alborz/personality-trait-predictor)

---

## License

This project is for academic and research use only.  
For non-academic use, please contact the author.

License: MIT

---

## Authors 
AMIV NLP 2025 — University of Antwerp