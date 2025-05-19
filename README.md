# Personality Trait Predictor — AMIV NLP 2025  
University of Antwerp

This project predicts **Big Five personality traits (OCEAN)** from English text using a combination of:

- DistilBERT embeddings  
- LIWC-style psycholinguistic features  
- An ensemble classifier (Random Forest, XGBoost, MLP, SVM)

The five traits predicted are:
- **Openness**
- **Conscientiousness**
- **Extraversion**
- **Agreeableness**
- **Emotional Stability**

Each trait is classified as:
- `low`
- `medium`
- `high`

---

## Features

- Accepts raw text (e.g., job interview answers)
- Extracts both semantic (BERT) and psycholinguistic (LIWC) features
- Outputs all 5 personality traits using a custom-trained ensemble
- Can be used locally or deployed via Gradio (demo available)

---

## Quick Usage (Python)

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
├── personality_model.py          # PersonalityClassifier pipeline
├── test_personality_model.py     # CLI tester
├── feature_extraction/
│   ├── __init__.py
│   ├── embedding_from_text.py
│   ├── liwc_from_text.py
├── models/
│   ├── openness_classifier.pkl
│   ├── conscientiousness_classifier.pkl
│   ├── extraversion_classifier.pkl
│   ├── agreeableness_classifier.pkl
│   ├── emotional_stability_classifier.pkl
│   ├── feature_scaler.pkl
│   ├── output.dic
├── requirements.txt
├── README.md
```

---

## Modeling Details

- Ensemble of 4 classifiers (VotingClassifier):
  - `RandomForestClassifier`
  - `GradientBoostingClassifier`
  - `MLPClassifier`
  - `SVC (linear)`
- Each trait has a separate classifier trained on combined BERT+LIWC features
- LIWC-style dictionary created from `output.dic`

---

## Preprocessing & Binning (for original experiments)

The original project also included regression models and binning rules:

| Score Range       | Bin Label |
|-------------------|-----------|
| 0 ≤ score ≤ 32    | Low       |
| 33 ≤ score ≤ 66   | Medium    |
| 67 ≤ score ≤ 100  | High      |

These were used to convert continuous personality scores into discrete labels.

---

## Evaluation Scripts

- Located in `evaluation/` folder (not shown here)
- Used during development to benchmark model performance
- Final classifiers are saved in `models/`

---

## Installation & Environment

Python: `3.9`  
Recommended: `conda` environment

```bash
conda create -n amiv_nlp_2025 python=3.9
conda activate amiv_nlp_2025
pip install -r requirements.txt
```

---

## License

For research and non-commercial use. Contact the author for other permissions.

---

## Authors

Developed by 
AMIV NLP 2025 — University of Antwerp  