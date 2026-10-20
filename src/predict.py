"""
src/predict.py
Inference module for Student Feedback Sentiment Prediction
Loads the trained best model and provides single-text and batch DataFrame prediction.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import clean_text

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


class SentimentPredictor:
    """Predictor class encapsulating model, vectorizer, and label encoder."""

    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.metadata = {}
        self.load_artifacts()

    def load_artifacts(self):
        """Load serialized model, TF-IDF vectorizer, and label encoder."""
        model_path = os.path.join(self.models_dir, "best_model.joblib")
        vec_path = os.path.join(self.models_dir, "tfidf_vectorizer.joblib")
        le_path = os.path.join(self.models_dir, "label_encoder.joblib")
        meta_path = os.path.join(self.models_dir, "model_metadata.json")

        if not (os.path.exists(model_path) and os.path.exists(vec_path) and os.path.exists(le_path)):
            raise FileNotFoundError(
                f"Model artifacts not found in {self.models_dir}. Please run 'python src/train.py' first."
            )

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vec_path)
        self.label_encoder = joblib.load(le_path)

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def predict_sentiment(self, text: str, aspect: str = "teaching") -> dict:
        """
        Predict sentiment and confidence score for a single feedback comment.
        Format: "<aspect>: <cleaned_text>"
        """
        if not text or not str(text).strip():
            return {
                "sentiment": "neutral",
                "confidence": 0.3333,
                "probabilities": {"negative": 0.3333, "neutral": 0.3334, "positive": 0.3333},
                "aspect": aspect,
                "clean_text": ""
            }

        aspect_norm = str(aspect).strip().lower()
        cleaned = clean_text(str(text))
        feature_text = f"{aspect_norm}: {cleaned}" if cleaned else f"{aspect_norm}: neutral"

        X_feat = self.vectorizer.transform([feature_text])

        # Get probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_feat)[0]
        elif hasattr(self.model, "decision_function"):
            df_vals = self.model.decision_function(X_feat)[0]
            exp_vals = np.exp(df_vals - np.max(df_vals))
            probs = exp_vals / np.sum(exp_vals)
        else:
            pred_idx = self.model.predict(X_feat)[0]
            probs = np.zeros(len(self.label_encoder.classes_))
            probs[pred_idx] = 1.0

        classes = self.label_encoder.classes_.tolist()
        pred_idx = int(np.argmax(probs))
        pred_label = classes[pred_idx]
        confidence = float(probs[pred_idx])

        prob_dict = {cls: round(float(probs[i]), 4) for i, cls in enumerate(classes)}

        return {
            "sentiment": pred_label,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "aspect": aspect_norm,
            "clean_text": cleaned
        }

    def predict_dataframe(self, df: pd.DataFrame, text_col: str, aspect_col: str = None) -> pd.DataFrame:
        """
        Batch prediction on a pandas DataFrame.
        Adds 'clean_text', 'predicted_sentiment', and 'confidence' columns.
        """
        res_df = df.copy()
        pred_sentiments = []
        confidences = []
        clean_texts = []

        for idx, row in res_df.iterrows():
            txt = row[text_col] if pd.notna(row[text_col]) else ""
            asp = row[aspect_col] if (aspect_col and aspect_col in row and pd.notna(row[aspect_col])) else "general"
            pred_res = self.predict_sentiment(str(txt), aspect=str(asp))
            
            clean_texts.append(pred_res["clean_text"])
            pred_sentiments.append(pred_res["sentiment"])
            confidences.append(pred_res["confidence"])

        res_df["clean_text"] = clean_texts
        res_df["predicted_sentiment"] = pred_sentiments
        res_df["confidence"] = confidences
        return res_df


# Module-level predictor singleton
_PREDICTOR = None


def get_predictor():
    """Retrieve or initialize the global predictor instance."""
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = SentimentPredictor()
    return _PREDICTOR


def predict_sentiment(text: str, aspect: str = "teaching") -> dict:
    """Convenience functional wrapper for single text prediction."""
    return get_predictor().predict_sentiment(text, aspect)


def predict_dataframe(df: pd.DataFrame, text_col: str, aspect_col: str = None) -> pd.DataFrame:
    """Convenience functional wrapper for batch DataFrame prediction."""
    return get_predictor().predict_dataframe(df, text_col, aspect_col)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict sentiment for student feedback")
    parser.add_argument("--text", type=str, default="The professor explains concepts thoroughly with practical examples.", help="Feedback comment")
    parser.add_argument("--aspect", type=str, default="teaching", help="Aspect name")
    args = parser.parse_args()

    result = predict_sentiment(args.text, args.aspect)
    print("\nPrediction Result:")
    print(json.dumps(result, indent=2))
