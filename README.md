#  **amiv_nlp_2025**
#  **University Of Antwerp**

This repository contains the code and scripts for the **Personality Prediction Project** using TF-IDF, Embeddings, and Regression Models.

The project includes:

Preprocessing text data (e.g. TF-IDF, embeddings)  
Training regression models (e.g. Ridge, Random Forest, XGBoost, Ensembles)  
Classification step based on regression outputs (binning → low, medium, high)  
Evaluation (Accuracy, F1-score, Classification Reports)

---

##  Getting Started

### Clone the Repository

```bash
git clone https://github.com/YourUsername/amiv_nlp_2025.git
cd amiv_nlp_2025
```

### Python Version

This project requires **Python 3.9**.  
Please make sure your Conda environment uses Python 3.9 to avoid compatibility issues.

```bash
conda create -n amiv_nlp_2025 python=3.9
conda activate amiv_nlp_2025
```

### Install Requirements

Install all the required Python packages:

```bash
pip install -r requirements.txt
```

(see `requirements.txt` for exact versions)

---

## Project Structure

```
├── preprocessing/
│   ├── embedding_DistilBERT.ipynb
│   ├── tfidf_vectorizer.ipynb
│
├── evaluation/
│   ├── ...
│
├── config.py
├── requirements.txt
├── README.md
```

---

## Preprocessing

- All data processing and feature extraction is done in the `preprocessing/` folder.
- Two main methods:
  - **TF-IDF vectorization**
  - **BERT/DistilBERT embeddings**

The output is stored as `.csv` files and later used in regression models.

---

##  Modeling 

Once the data is preprocessed:

- Train **Regression Models** (Ridge, RandomForest, XGBoost, Ensemble)
- Predict Big Five personality scores as continuous values
- Apply **Binning function** to convert continuous scores to categories (Low, Medium, High)
- Evaluate with:
  - Accuracy
  - F1-score
  - Classification Report

All evaluation and modeling scripts are in `evaluation/` folder.

## Binning Rules

The personality trait scores (0–100) are converted into categorical labels using the following rules:

| Score Range | Bin Label |
|-------------|-----------|
| 0 ≤ score ≤ 32 | Low |
| 33 ≤ score ≤ 66 | Medium |
| 67 ≤ score ≤ 100 | High |

**Explanation:**

- If the score is between **0 and 32 (inclusive)** → Label: `Low`
- If the score is between **33 and 66 (inclusive)** → Label: `Medium`
- If the score is between **67 and 100 (inclusive)** → Label: `High`


You can use this function to assign bin labels to any numerical personality score.

---

## Notes and Best Practices

- Large datasets (TF-IDF `.csv`, raw `.json` etc.) should be **ignored using `.gitignore`** and should NOT be pushed to GitHub.
- Configurations (paths, parameters) should go into `config.py` to keep notebooks/scripts clean.
- Recommended Conda usage → easier reproducibility.

---

## Contact

For questions and contributions → open an issue or contact amiv_nlp_2025 University of Antwerp