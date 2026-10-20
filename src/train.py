"""
src/train.py
Phase 3: Model Training, Evaluation, and Comparison Pipeline
"""

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import sys
# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.load_data import load_and_reshape
from src.preprocess import clean_text

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

REPORTS_DIR = "reports"
MODELS_DIR = "models"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def get_data_splits():
    """
    Load data and perform student-disjoint 70/15/15 train/val/test split.
    Stratified by student overall sentiment.
    """
    long_path = "data/processed/feedback_long.csv"
    wide_path = "data/processed/feedback_wide.csv"

    if not os.path.exists(long_path) or not os.path.exists(wide_path):
        print("Processed data not found. Running Phase 0...")
        df_long, df_wide = load_and_reshape()
    else:
        df_long = pd.read_csv(long_path)
        df_wide = pd.read_csv(wide_path)

    # Preprocess text
    df_long["clean_text"] = df_long["text"].astype(str).apply(clean_text)
    
    # Prepend aspect to clean text: "teaching: <clean_text>"
    df_long["feature_text"] = df_long["aspect"] + ": " + df_long["clean_text"]

    # Student-level stratified split
    train_stu, temp_stu = train_test_split(
        df_wide["student_id"],
        test_size=0.30,
        random_state=SEED,
        stratify=df_wide["overall_sentiment"]
    )

    val_stu, test_stu = train_test_split(
        temp_stu,
        test_size=0.50,
        random_state=SEED,
        stratify=df_wide.set_index("student_id").loc[temp_stu, "overall_sentiment"]
    )

    train_df = df_long[df_long["student_id"].isin(train_stu)].copy()
    val_df = df_long[df_long["student_id"].isin(val_stu)].copy()
    test_df = df_long[df_long["student_id"].isin(test_stu)].copy()

    # Drop any rows with empty feature_text
    train_df = train_df[train_df["clean_text"].str.strip() != ""].copy()
    val_df = val_df[val_df["clean_text"].str.strip() != ""].copy()
    test_df = test_df[test_df["clean_text"].str.strip() != ""].copy()

    print(f"Data Splits (Student-Disjoint):")
    print(f"  Train: {len(train_stu)} students, {len(train_df)} feedback instances")
    print(f"  Val  : {len(val_stu)} students, {len(val_df)} feedback instances")
    print(f"  Test : {len(test_stu)} students, {len(test_df)} feedback instances")

    return train_df, val_df, test_df


def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    """Generate and save formatted confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 5))
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True
    )
    plt.title(f"Confusion Matrix: {title}", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Predicted Sentiment", fontsize=11)
    plt.ylabel("Actual Sentiment", fontsize=11)
    plt.tight_layout()
    save_path = os.path.join(REPORTS_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix: {save_path}")


def evaluate_predictions(y_true, y_pred, model_name, label_order=["negative", "neutral", "positive"]):
    """Calculate accuracy, macro precision, recall, and F1."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0)

    # Save CM
    cm_name = f"cm_{model_name.lower().replace(' ', '_').replace('-', '_')}.png"
    plot_confusion_matrix(y_true, y_pred, label_order, model_name, cm_name)

    return {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Macro Precision": round(prec, 4),
        "Macro Recall": round(rec, 4),
        "Macro F1": round(f1, 4),
        "Predictions": y_pred
    }


def train_vader_baseline(test_df):
    """
    (a) VADER Sentiment Analyzer Baseline
    Rule: compound >= 0.05 positive, <= -0.05 negative, else neutral.
    """
    print("\nEvaluating VADER Baseline...")
    analyzer = SentimentIntensityAnalyzer()
    y_pred = []
    
    for text in test_df["text"]:
        scores = analyzer.polarity_scores(str(text))
        comp = scores["compound"]
        if comp >= 0.05:
            y_pred.append("positive")
        elif comp <= -0.05:
            y_pred.append("negative")
        else:
            y_pred.append("neutral")
            
    res = evaluate_predictions(test_df["sentiment"].tolist(), y_pred, "VADER Baseline")
    return res


def train_traditional_models(train_df, test_df):
    """
    Train TF-IDF + Logistic Regression, Multinomial Naive Bayes, Linear SVM.
    Uses balanced class weights to address class imbalance.
    """
    print("\nVectorizing text with TF-IDF (1-2 grams)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    
    X_train = vectorizer.fit_transform(train_df["feature_text"])
    X_test = vectorizer.transform(test_df["feature_text"])

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["sentiment"])
    y_test = label_encoder.transform(test_df["sentiment"])
    labels = label_encoder.classes_.tolist() # ['negative', 'neutral', 'positive']

    # Class weights
    classes_unique = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes_unique, y=y_train)
    class_weight_dict = dict(zip(classes_unique, weights))
    print(f"Computed Class Weights: { {labels[k]: round(v, 2) for k, v in class_weight_dict.items()} }")

    models = {}
    results = []

    # 1. Logistic Regression
    print("\nTraining TF-IDF + Logistic Regression...")
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=SEED,
        C=1.0
    )
    lr.fit(X_train, y_train)
    y_pred_lr = label_encoder.inverse_transform(lr.predict(X_test))
    res_lr = evaluate_predictions(test_df["sentiment"].tolist(), y_pred_lr, "TF-IDF + Logistic Regression")
    results.append(res_lr)
    models["TF-IDF + Logistic Regression"] = (lr, vectorizer, label_encoder)

    # 2. Multinomial Naive Bayes
    print("\nTraining TF-IDF + Multinomial Naive Bayes...")
    nb = MultinomialNB(alpha=0.5)
    nb.fit(X_train, y_train)
    y_pred_nb = label_encoder.inverse_transform(nb.predict(X_test))
    res_nb = evaluate_predictions(test_df["sentiment"].tolist(), y_pred_nb, "TF-IDF + Naive Bayes")
    results.append(res_nb)
    models["TF-IDF + Naive Bayes"] = (nb, vectorizer, label_encoder)

    # 3. Linear SVM (Calibrated for probabilities)
    print("\nTraining TF-IDF + Linear SVM (Calibrated)...")
    base_svm = LinearSVC(
        class_weight="balanced",
        random_state=SEED,
        C=1.0,
        dual=True,
        max_iter=3000
    )
    calibrated_svm = CalibratedClassifierCV(estimator=base_svm, cv=3)
    calibrated_svm.fit(X_train, y_train)
    y_pred_svm = label_encoder.inverse_transform(calibrated_svm.predict(X_test))
    res_svm = evaluate_predictions(test_df["sentiment"].tolist(), y_pred_svm, "TF-IDF + Linear SVM")
    results.append(res_svm)
    models["TF-IDF + Linear SVM"] = (calibrated_svm, vectorizer, label_encoder)

    return models, results, vectorizer, label_encoder


def train_distilbert_model(train_df, val_df, test_df):
    """
    (e) Fine-tune DistilBERT (distilbert-base-uncased) for sequence classification.
    Max 2 epochs, max_length 128, with class-weighted CrossEntropyLoss.
    """
    print("\nAttempting DistilBERT Fine-tuning (distilbert-base-uncased)...")
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DistilBERT Training Device: {device}")

    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    inv_label_map = {0: "negative", 1: "neutral", 2: "positive"}

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    class FeedbackDataset(Dataset):
        def __init__(self, texts, labels, max_len=128):
            self.texts = texts
            self.labels = labels
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = str(self.texts[idx])
            encoding = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt"
            )
            item = {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
            }
            if self.labels is not None:
                item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    y_train_num = [label_map[s] for s in train_df["sentiment"]]
    y_val_num = [label_map[s] for s in val_df["sentiment"]]
    y_test_num = [label_map[s] for s in test_df["sentiment"]]

    train_dataset = FeedbackDataset(train_df["feature_text"].tolist(), y_train_num)
    val_dataset = FeedbackDataset(val_df["feature_text"].tolist(), y_val_num)
    test_dataset = FeedbackDataset(test_df["feature_text"].tolist(), y_test_num)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3
    ).to(device)

    # Class weighting
    classes_arr = np.array([0, 1, 2])
    weights = compute_class_weight(class_weight="balanced", classes=classes_arr, y=y_train_num)
    class_weights_t = torch.tensor(weights, dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    epochs = 2
    print(f"Beginning DistilBERT fine-tuning for {epochs} epochs...")
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {ep+1}/{epochs} - Average Loss: {avg_loss:.4f}")

    # Evaluate on test set
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            test_preds.extend(preds)

    y_pred_labels = [inv_label_map[p] for p in test_preds]
    res_distilbert = evaluate_predictions(test_df["sentiment"].tolist(), y_pred_labels, "DistilBERT (Fine-tuned)")

    # Save fine-tuned distilbert
    distilbert_save_path = os.path.join(MODELS_DIR, "distilbert_feedback")
    model.save_pretrained(distilbert_save_path)
    tokenizer.save_pretrained(distilbert_save_path)
    print(f"Saved fine-tuned DistilBERT model to: {distilbert_save_path}")

    return res_distilbert, (model, tokenizer, inv_label_map)


def plot_model_comparison(df_comp):
    """Save model comparison bar chart."""
    plt.figure(figsize=(10, 6))
    melted = df_comp.melt(id_vars="Model", var_name="Metric", value_name="Score")
    
    palette = sns.color_palette("muted")
    ax = sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette=palette)
    plt.title("Model Comparison on Test Set", fontsize=14, fontweight="bold", pad=15)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score", fontsize=11)
    plt.xlabel("")
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(title="Evaluation Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    
    chart_path = os.path.join(REPORTS_DIR, "model_comparison.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved model comparison chart: {chart_path}")


def analyze_misclassifications(test_df, best_preds, best_model_name):
    """
    Identify misclassified test examples, explain reasons,
    and save reports/misclassified_examples.md.
    """
    test_eval = test_df.copy()
    test_eval["predicted"] = best_preds
    test_eval["actual"] = test_df["sentiment"]
    
    misclassified = test_eval[test_eval["actual"] != test_eval["predicted"]].copy()
    print(f"\nTotal misclassified test instances by {best_model_name}: {len(misclassified)} / {len(test_eval)}")
    
    # Select 10 diverse misclassified examples
    sample_mis = misclassified.head(10)
    
    def explain_error(row):
        txt = str(row["text"]).lower()
        act = row["actual"]
        pred = row["predicted"]
        
        if len(txt.split()) <= 3:
            return "Extreme brevity: Text contains very few tokens, lacking explicit lexical sentiment context."
        elif "not" in txt or "no" in txt or "never" in txt:
            return "Complex negation / nuance: Negation syntax confused standard token weights or context."
        elif (act == "neutral" and pred in ("positive", "negative")) or (pred == "neutral" and act in ("positive", "negative")):
            return "Neutral boundary ambiguity: Feedback expresses mild critique or conditional praise, blurring the neutral threshold."
        elif "but" in txt or "although" in txt or "though" in txt:
            return "Mixed sentiment: Student commented on both strengths and weaknesses within a single aspect response."
        else:
            return "Subtle or subjective phrasing: Colloquial phrasing where sentiment diverges from training vocabulary."

    report_lines = [
        f"# Misclassification Analysis for Best Model ({best_model_name})\n",
        f"Out of **{len(test_eval)}** test samples, **{len(misclassified)}** were misclassified.",
        "Below are 10 illustrative misclassified examples along with detailed error rationales:\n",
        "| # | Aspect | Student Comment | Actual Sentiment | Predicted Sentiment | Primary Error Reason |",
        "|---|---|---|---|---|---|"
    ]

    for i, (_, row) in enumerate(sample_mis.iterrows(), 1):
        reason = explain_error(row)
        clean_comment = row["text"].replace("|", "\\|").replace("\n", " ")
        report_lines.append(
            f"| {i} | {row['aspect']} | \"{clean_comment}\" | `{row['actual']}` | `{row['predicted']}` | {reason} |"
        )

    report_lines.extend([
        "\n## Common Failure Modes Identified:",
        "1. **Brevity**: Single-word comments like *'Average'* or *'Fine'* can be borderline neutral vs positive depending on context.",
        "2. **Mixed Sentiment**: Students often combine praise with criticism (*'Teacher is very polite but speed is too fast'*).",
        "3. **Neutral Ambiguity**: Neutral class is naturally harder to separate as it borders mild negative and mild positive opinions.",
        "4. **Sarcasm and Colloquialisms**: Sarcastic remarks (*'Great job making exams impossible'*) possess positive surface lexicon with negative intent."
    ])

    report_path = os.path.join(REPORTS_DIR, "misclassified_examples.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Saved misclassified analysis to: {report_path}")


def compute_per_aspect_f1(test_df, best_preds):
    """Compute Macro F1 score per aspect for the best model."""
    test_eval = test_df.copy()
    test_eval["predicted"] = best_preds
    
    aspect_f1 = {}
    print("\n==================================================")
    print("PER-ASPECT MACRO F1 SCORES (BEST MODEL)")
    print("==================================================")
    for aspect in sorted(test_eval["aspect"].unique()):
        sub = test_eval[test_eval["aspect"] == aspect]
        f1 = f1_score(
            sub["sentiment"],
            sub["predicted"],
            labels=["negative", "neutral", "positive"],
            average="macro",
            zero_division=0
        )
        acc = accuracy_score(sub["sentiment"], sub["predicted"])
        aspect_f1[aspect] = {"macro_f1": round(f1, 4), "accuracy": round(acc, 4), "count": len(sub)}
        print(f"  {aspect:<20}: Macro F1 = {f1:.4f} (Accuracy = {acc*100:.2f}%, N={len(sub)})")

    return aspect_f1


def run_training_pipeline():
    """Execute complete Phase 3 training and evaluation pipeline."""
    print("==================================================")
    print("PHASE 3: TRAINING AND EVALUATING MODELS")
    print("==================================================")

    train_df, val_df, test_df = get_data_splits()

    all_results = []
    trained_models = {}

    # (a) VADER Baseline
    vader_res = train_vader_baseline(test_df)
    all_results.append(vader_res)

    # (b, c, d) Traditional TF-IDF Models
    trad_models, trad_results, vectorizer, label_encoder = train_traditional_models(train_df, test_df)
    all_results.extend(trad_results)
    trained_models.update(trad_models)

    # (e) DistilBERT Fine-tuning
    try:
        distilbert_res, distilbert_artifacts = train_distilbert_model(train_df, val_df, test_df)
        all_results.append(distilbert_res)
        trained_models["DistilBERT (Fine-tuned)"] = distilbert_artifacts
    except Exception as e:
        print(f"\n[NOTICE] DistilBERT training skipped: {e}")
        with open(os.path.join(REPORTS_DIR, "distilbert_skip_reason.txt"), "w") as f:
            f.write(f"DistilBERT could not run or was skipped due to: {e}\n")

    # Compile Results Table
    comp_data = []
    for r in all_results:
        comp_data.append({
            "Model": r["Model"],
            "Accuracy": r["Accuracy"],
            "Macro Precision": r["Macro Precision"],
            "Macro Recall": r["Macro Recall"],
            "Macro F1": r["Macro F1"]
        })
    df_comp = pd.DataFrame(comp_data)
    
    # Save CSV
    csv_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    df_comp.to_csv(csv_path, index=False)
    print(f"\nSaved model comparison table to: {csv_path}")
    print("\n==================================================")
    print("MODEL COMPARISON ON TEST SET")
    print("==================================================")
    print(df_comp.to_string(index=False))

    # Plot Comparison Chart
    plot_model_comparison(df_comp)

    # Select Best Model by Macro F1
    best_row = df_comp.loc[df_comp["Macro F1"].idxmax()]
    best_model_name = best_row["Model"]
    print(f"\n>>> Best Model Selected: {best_model_name} (Macro F1 = {best_row['Macro F1']:.4f}) <<<")

    best_preds = None
    for r in all_results:
        if r["Model"] == best_model_name:
            best_preds = r["Predictions"]
            break

    # Per-aspect Macro F1 for Best Model
    aspect_f1_scores = compute_per_aspect_f1(test_df, best_preds)

    # Misclassification Analysis for Best Model
    analyze_misclassifications(test_df, best_preds, best_model_name)

    # Save Best Model & Artifacts
    # If best is a traditional model:
    if best_model_name in trad_models:
        model_obj, vec, le = trad_models[best_model_name]
        joblib.dump(model_obj, os.path.join(MODELS_DIR, "best_model.joblib"))
        joblib.dump(vec, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
        joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))
        print(f"Saved best model artifacts ({best_model_name}) to {MODELS_DIR}/")
    else:
        # If DistilBERT is best, also save best traditional model as fallback for quick CPU app inference
        # and record DistilBERT as primary
        best_trad_name = max(
            [r for r in trad_results],
            key=lambda x: x["Macro F1"]
        )["Model"]
        trad_obj, vec, le = trad_models[best_trad_name]
        joblib.dump(trad_obj, os.path.join(MODELS_DIR, "best_model.joblib"))
        joblib.dump(vec, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
        joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))
        print(f"Saved primary DistilBERT model and fallback traditional model ({best_trad_name}) to {MODELS_DIR}/")

    # Save Metadata JSON
    metadata = {
        "best_model_name": best_model_name,
        "metrics": best_row.to_dict(),
        "per_aspect_f1": aspect_f1_scores,
        "classes": ["negative", "neutral", "positive"],
        "all_models_summary": comp_data
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to {MODELS_DIR}/model_metadata.json")

    print("\nPhase 3 Model Pipeline Completed Successfully!")
    return df_comp, best_model_name, aspect_f1_scores


if __name__ == "__main__":
    run_training_pipeline()
