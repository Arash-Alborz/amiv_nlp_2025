from personality_model import PersonalityClassifier

# avoinding transformer warnings
import warnings
import logging
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# loading the classifier
model = PersonalityClassifier()

# example
text = """
I’ve always been fascinated by different cultures and philosophies. I love reading poetry, exploring new ideas, and reflecting on the complexity of human emotion. 
Traveling, learning languages, and experiencing art open up new perspectives for me every time.
"""

# predictions
results = model.predict_all_traits(text)

print("\nPredicted Personality Traits:")
for trait, label in results.items():
    print(f"{trait}: {label}")