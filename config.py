'''
Change the path to your files.

'''

# Path to your folder where the datasets are
DATA_FOLDER = "/Users/arashalborz/Desktop/amiv_nlp_2025/Data"

# Specific datasets, filenames etc.
TRAIN = "filtered_pandora.json"
VALIDATION_VALUES = "val_data_realvalued.csv"
VALIDATION_LABELS = "val_data.csv"


PANDORA_COMMENTS = "all_comments_since_2015.csv"
PANDORA_COMMENTS = "author_profiles.csv"


# config.py

# Data paths
DATA_DIR = "data"
RESULTS_DIR = "results"
REPORTS_DIR = "results/classification_reports"

# TF-IDF
TFIDF_MAX_FEATURES = 1000

# Binning
BINNING_THRESHOLDS = [32, 66]
BINNING_LABELS = ["Low", "Medium", "High"]