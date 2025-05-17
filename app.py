# app.py

import gradio as gr
import joblib
import numpy as np
from feature_extraction.pipeline import text_to_features

# Load pretrained Random Forest model for Openness
model = joblib.load("models/openness_rf.pkl")

def predict_openness(text):
    try:
        vec = text_to_features(text)  # shape: (1, dim)
        pred = model.predict(vec)[0]  # already "low", "medium", or "high"
        return f"Predicted Openness: **{pred.upper()}**"
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio UI
demo = gr.Interface(
    fn=predict_openness,
    inputs=gr.Textbox(lines=6, placeholder="Enter your thoughts here..."),
    outputs=gr.Markdown(),
    title="Big Five Personality Prediction",
    description="This model predicts **Openness** based on your text using BERT + LIWC features.",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()