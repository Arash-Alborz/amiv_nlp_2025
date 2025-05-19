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

IMPORTANT: For testing the model use 
This repository contains all components of the project:

- **preprocessingn/**: Code for extracting BERT embeddings and LIWC-style features / Scripts for binning continuous scores, cleaning, and formatting raw data
- **classification/**: Training, evaluation, and grid search scripts for different classifiers
- **synthetic_data_generation/**: Scripts for generating synthetic job interview-style responses using OpenAI GPT
- **data_analysis/**: Tools for comparing classifiers, saving performance reports, and generating visual summaries
- **processed_data/**: liwc features and embeddings are available as csv files for training and further optimization
- **personality_model/**: A separate folder containing the deployable final model

---

## Quick Usage Example (Prediction)

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
├── data_analysis/
    └── statistics.ipynb    # getting statistics for text length, label distribution (train & val)
├── preprocessing/          # scripts for extracting tf-idt/embedding/liwc/ etc. / cleaning data
    ├── merge_files.ipynb
    ├── converting_numbers_to_labels.ipynb
    ├── embedding_DistillBERT.ipynb
    ├── liwc_features.ipynb
    ├── output.dict         # for mapping liwc
    ├── merge_synthetic.ipynb 
    └── tfidf_vectorizer.ipynb
├── processed_data/
    ├── train               # containing csv files with embeddings/tfidf/liwc & embeding+liwc
    └──  validation          # containing csv files with embeddings/tfidf/liwc & embeding+liwc
├── synthetic_data_generation/
    ├── synthetic_data_generation.ipynb   # script for generating data with GPT
    └── synthetic_val_data.csv
├── classification/
    ├── classification_task.ipynb   # ML algorithms trained on train/val 
    ├── random_forest.ipynb         # optimizing random forest classifier
    ├── reports                     # all classification reports in .txt format
      └── rf_reports                # classification report for optimization of rf
  ├── personality_model/            # final model 
    ├── feature_extraction/
      ├── __init__.py
      ├── embedding_from_text.py    # getting embedding from raw text
      └── liwc_from_text.py         # getting liwc features from raw text
    ├── models/                     # pretrained rf-classifiers for each trait
      ├── agreeableness_classifier.pkl
      ├── openness_classifier.pkl
      ├── emotional_stability_classifier.pkl
      ├── extraversion_classifier.pkl
      ├── conscientiousness_classifier.pkl
      ├── feature_scaler.pkl        # standardscaler() used in class
      └── output_dict               # for getting liwc features from raw text
    ├── personality_model.py        # initializing the classifiers
    ├── predoct_from_csv.ipynb      # CODE THAT CAN BE USED FOR TEST DATA
    └── test_personality_model      # code for testing the model on raw text
├── .gitattributes
├── .gitignore
├── README.dm
├── requirements.txt
└── saving_pretrained_models.ipynb  # for freezing the trained classifiers

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
conda create -n personality_env python=3.9
conda activate personality_env

# Install dependencies
pip install -r requirements.txt
```

---


## Local Testing

Run quick tests on raw text:
```bash
python test_personality_model.py
```

Or run predictions on a CSV file of interview responses (Q1, Q2, Q3):
```bash
jupyter notebook predict_from_csv.ipynb
```

## Hugging Face Model

A clean version of the model (ready for deployment, withouht other codes) is stored in `hf_personality_model/`. This version contains:
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