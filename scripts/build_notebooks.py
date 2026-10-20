"""
scripts/build_notebooks.py
Generates and executes notebooks/01_eda.ipynb and notebooks/02_training_evaluation.ipynb
with full outputs, embedded figures, and simple English markdown explanations.
"""

import os
import sys
import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)


def build_eda_notebook():
    """Create and execute 01_eda.ipynb"""
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# 01. Exploratory Data Analysis (EDA)
## Student Feedback Analysis using NLP (Aspect-Based Sentiment Analysis)

### Objective
This notebook performs exploratory analysis on the student feedback dataset:
1. Loading the raw wide spreadsheet (`student-data.xlsx`)
2. Inspecting column names, shapes, data types, missing values, and duplicates
3. Reshaping from WIDE to LONG aspect format
4. Analyzing class distributions (overall and per-aspect)
5. Examining comment length distributions across sentiment categories
6. Identifying frequent vocabulary and keywords
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
for p in [".", ".."]:
    abs_p = os.path.abspath(p)
    if os.path.exists(os.path.join(abs_p, "src")) and abs_p not in sys.path:
        sys.path.insert(0, abs_p)

from src.preprocess import clean_text

# Plotting configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
%matplotlib inline
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 1. Load Raw Dataset & Diagnostics
The raw survey contains student ratings and open-ended feedback across 6 core aspects:
- `teaching` + `teachingl`
- `coursecontent` + `coursecontentl`
- `examination` + `examinationl`
- `labwork` + `labworkl`
- `library_facilities` + `library_facilitiesl`
- `extracurricular` + `extracurricularl`
"""))

    cells.append(nbf.v4.new_code_cell("""# Locate dataset file dynamically
candidates = [
    "student-data.xlsx",
    "../student-data.xlsx",
    "data/raw/student-data.xlsx",
    "../data/raw/student-data.xlsx"
]
data_file = next(os.path.abspath(c) for c in candidates if os.path.exists(c))

df_raw = pd.read_excel(data_file)
print(f"Dataset File : {os.path.basename(data_file)}")
print(f"Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
print("\\nDetected Columns:")
for c in df_raw.columns:
    print(f"  - '{c}' ({df_raw[c].dtype})")
"""))

    cells.append(nbf.v4.new_code_cell("""# Check missing values and duplicate records
print("Missing values per column:")
print(df_raw.isnull().sum())

print(f"\\nDuplicate survey responses in raw data: {df_raw.duplicated().sum()}")
df_raw.head(3)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2. Reshaping WIDE to LONG Format
To train aspect-based sentiment models, we reshape the wide dataset into a LONG table where each row corresponds to one `(student_id, aspect, text, label)`.
We also compute an `overall_sentiment` per student from their average aspect label:
- $\\text{mean} \\le -0.34 \\rightarrow \\text{negative}$
- $\\text{mean} \\ge 0.34 \\rightarrow \\text{positive}$
- $\\text{otherwise} \\rightarrow \\text{neutral}$
"""))

    cells.append(nbf.v4.new_code_cell("""# Locate processed data
long_cands = ["data/processed/feedback_long.csv", "../data/processed/feedback_long.csv"]
wide_cands = ["data/processed/feedback_wide.csv", "../data/processed/feedback_wide.csv"]

long_path = next(os.path.abspath(c) for c in long_cands if os.path.exists(c))
wide_path = next(os.path.abspath(c) for c in wide_cands if os.path.exists(c))

df_long = pd.read_csv(long_path)
df_wide = pd.read_csv(wide_path)

print(f"Long Format Rows (Feedback Instances): {len(df_long)}")
print(f"Wide Format Rows (Students)           : {len(df_wide)}")
df_long.sample(5, random_state=42)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3. Overall Class Distribution
Let us examine the balance between Positive, Neutral, and Negative feedback comments across the entire dataset.
"""))

    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(7, 4))
colors = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}
dist = df_long['sentiment'].value_counts()
pcts = df_long['sentiment'].value_counts(normalize=True) * 100

bars = plt.bar(dist.index.str.capitalize(), dist.values, color=[colors[c] for c in dist.index], edgecolor='black')
for bar in bars:
    h = bar.get_height()
    pct = (h / len(df_long)) * 100
    plt.text(bar.get_x() + bar.get_width()/2, h + 15, f"{h} ({pct:.1f}%)", ha='center', fontweight='bold')

plt.title("Overall Sentiment Distribution (1,324 Comments)", fontsize=13, fontweight='bold')
plt.ylabel("Number of Comments")
plt.ylim(0, max(dist.values) + 80)
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 4. Per-Aspect Class Distribution
Next, we evaluate how sentiment varies across each of the six academic aspects.
"""))

    cells.append(nbf.v4.new_code_cell("""crosstab = pd.crosstab(df_long['aspect'], df_long['sentiment'], normalize='index')[['positive', 'neutral', 'negative']] * 100

ax = crosstab.plot(kind='barh', stacked=True, color=['#2ecc71', '#95a5a6', '#e74c3c'], figsize=(10, 5), edgecolor='black')
plt.title("Aspect-Wise Sentiment Proportions (%)", fontsize=13, fontweight='bold')
plt.xlabel("Percentage (%)")
plt.ylabel("Evaluation Aspect")
plt.legend(["Positive", "Neutral", "Negative"], loc="lower right")
plt.xlim(0, 100)
plt.show()

print("Percentage Breakdown per Aspect:")
display(crosstab.round(1))
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 5. Comment Length Analysis by Sentiment
Are negative comments systematically longer than positive comments? Let us measure word and character lengths.
"""))

    cells.append(nbf.v4.new_code_cell("""df_long['char_count'] = df_long['text'].astype(str).str.len()
df_long['word_count'] = df_long['text'].astype(str).str.split().str.len()

print("Length Summary by Sentiment Category:")
display(df_long.groupby('sentiment')[['word_count', 'char_count']].agg(['mean', 'median', 'std']).round(1))

plt.figure(figsize=(8, 4))
sns.boxplot(data=df_long, x='sentiment', y='word_count', palette=['#2ecc71', '#95a5a6', '#e74c3c'], showmeans=True)
plt.title("Word Count Distribution by Sentiment", fontsize=13, fontweight='bold')
plt.xlabel("Sentiment")
plt.ylabel("Words per Comment")
plt.ylim(0, 45)
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 6. Student-Level Overall Sentiment
Distribution of overall student sentiment computed from their aspect score average:
"""))

    cells.append(nbf.v4.new_code_cell("""stu_dist = df_wide['overall_sentiment'].value_counts()
print("Student-Level Sentiment Distribution:")
for s, c in stu_dist.items():
    print(f"  {s.capitalize():<10}: {c} students ({c/len(df_wide)*100:.1f}%)")
"""))

    nb.cells = cells
    nb_path = os.path.join(NOTEBOOKS_DIR, "01_eda.ipynb")
    
    print(f"Executing and saving {nb_path}...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={'metadata': {'path': PROJECT_ROOT}})
    client.execute()
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("01_eda.ipynb successfully generated and executed with saved outputs!")


def build_training_notebook():
    """Create and execute 02_training_evaluation.ipynb"""
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# 02. Model Training & Evaluation Pipeline
## Student Feedback Analysis using NLP (Aspect-Based Sentiment Analysis)

### Pipeline Overview
This notebook presents the complete machine learning and NLP pipeline:
1. Text Preprocessing (negation preservation, contraction expansion, lemmatization)
2. Student-Disjoint Stratified Split (70% Train, 15% Val, 15% Test)
3. Model 1: VADER Rule-Based Baseline
4. Model 2: TF-IDF (1-2 grams) + Logistic Regression (Class-weighted)
5. Model 3: TF-IDF + Multinomial Naive Bayes
6. Model 4: TF-IDF + Linear SVM (Calibrated)
7. Model 5: Fine-Tuned DistilBERT (Contextual Transformer)
8. Comprehensive Benchmark Comparison and Error Analysis
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is accessible
for p in [".", ".."]:
    abs_p = os.path.abspath(p)
    if os.path.exists(os.path.join(abs_p, "src")) and abs_p not in sys.path:
        sys.path.insert(0, abs_p)

from src.preprocess import clean_text
from src.predict import get_predictor

%matplotlib inline
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 1. Text Preprocessing Demonstration
Negations are critical for sentiment. Our custom `clean_text` function strips noise while strictly preserving negation tokens (`not`, `no`, `never`, `n't`) and lemmatizing root words.
"""))

    cells.append(nbf.v4.new_code_cell("""demo_comments = [
    "The teacher is NOT explaining concepts clearly at all!",
    "I don't like the examination pattern, it's very difficult.",
    "Excellent lab workstations and helpful assistants."
]

print("Preprocessing Demonstration:")
for comment in demo_comments:
    print(f"Original: {comment}")
    print(f"Cleaned : {clean_text(comment)}\\n")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2. Student-Disjoint Train/Val/Test Split
To ensure zero data leakage between training and testing, feedback comments from the same `student_id` are strictly isolated to either Train, Val, or Test.
"""))

    cells.append(nbf.v4.new_code_cell("""# Locate processed files
long_cands = ["data/processed/feedback_long.csv", "../data/processed/feedback_long.csv"]
wide_cands = ["data/processed/feedback_wide.csv", "../data/processed/feedback_wide.csv"]

long_path = next(os.path.abspath(c) for c in long_cands if os.path.exists(c))
wide_path = next(os.path.abspath(c) for c in wide_cands if os.path.exists(c))

df_long = pd.read_csv(long_path)
df_wide = pd.read_csv(wide_path)

from sklearn.model_selection import train_test_split

train_stu, temp_stu = train_test_split(
    df_wide['student_id'],
    test_size=0.30,
    random_state=42,
    stratify=df_wide['overall_sentiment']
)

val_stu, test_stu = train_test_split(
    temp_stu,
    test_size=0.50,
    random_state=42,
    stratify=df_wide.set_index('student_id').loc[temp_stu, 'overall_sentiment']
)

train_df = df_long[df_long['student_id'].isin(train_stu)]
test_df = df_long[df_long['student_id'].isin(test_stu)]

print(f"Train Cohort: {len(train_stu)} students ({len(train_df)} feedback instances)")
print(f"Test Cohort : {len(test_stu)} students ({len(test_df)} feedback instances)")
assert len(set(train_stu).intersection(set(test_stu))) == 0, "Error: Student leakage detected!"
print("Verified: Zero student overlap between train and test!")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3. Model Benchmark Comparison
All five models were trained and evaluated on the exact same student-disjoint test set.
"""))

    cells.append(nbf.v4.new_code_cell("""comp_cands = ["reports/model_comparison.csv", "../reports/model_comparison.csv"]
comp_path = next(os.path.abspath(c) for c in comp_cands if os.path.exists(c))
df_comp = pd.read_csv(comp_path)
print("Model Benchmark Summary on Test Set:")
display(df_comp)
"""))

    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(10, 5))
melted = df_comp.melt(id_vars="Model", var_name="Metric", value_name="Score")
sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette="muted")
plt.title("Model Comparison Across Test Metrics", fontsize=13, fontweight='bold')
plt.ylim(0, 1.05)
plt.xticks(rotation=15, ha='right')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 4. Per-Aspect Performance of the Best Model
Let us examine the performance of **TF-IDF + Logistic Regression** broken down across each individual aspect.
"""))

    cells.append(nbf.v4.new_code_cell("""meta_cands = ["models/model_metadata.json", "../models/model_metadata.json"]
meta_path = next(os.path.abspath(c) for c in meta_cands if os.path.exists(c))
with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

aspect_f1 = meta["per_aspect_f1"]
df_asp_f1 = pd.DataFrame(aspect_f1).T.reset_index()
df_asp_f1.columns = ["Aspect", "Macro F1", "Accuracy", "Test Instances"]
display(df_asp_f1)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 5. Confusion Matrix (Best Model)
"""))

    cells.append(nbf.v4.new_code_cell("""from PIL import Image
cm_cands = ["reports/cm_tf_idf_+_logistic_regression.png", "../reports/cm_tf_idf_+_logistic_regression.png"]
cm_img_path = next((os.path.abspath(c) for c in cm_cands if os.path.exists(c)), None)
if cm_img_path:
    img = Image.open(cm_img_path)
    plt.figure(figsize=(6, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Confusion Matrix: TF-IDF + Logistic Regression", fontsize=12, fontweight='bold')
    plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 6. Misclassification Analysis
10 diverse misclassified examples and error rationales:
"""))

    cells.append(nbf.v4.new_code_cell("""mis_cands = ["reports/misclassified_examples.md", "../reports/misclassified_examples.md"]
mis_path = next(os.path.abspath(c) for c in mis_cands if os.path.exists(c))
with open(mis_path, "r", encoding="utf-8") as f:
    print(f.read())
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 7. Live Model Inference Test
Demonstrating real-time prediction using the loaded best model pipeline:
"""))

    cells.append(nbf.v4.new_code_cell("""predictor = get_predictor()

test_cases = [
    ("The lab computers are very slow and need urgent repair.", "labwork"),
    ("Professors explain concepts thoroughly with practical examples.", "teaching"),
    ("The syllabus is okay, just standard topics.", "coursecontent")
]

for text, aspect in test_cases:
    pred = predictor.predict_sentiment(text, aspect=aspect)
    print(f"Aspect : {aspect}")
    print(f"Comment: \\"{text}\\"")
    print(f"Result : {pred['sentiment'].upper()} (Confidence: {pred['confidence']*100:.1f}%)")
    print(f"Probs  : {pred['probabilities']}\\n")
"""))

    nb.cells = cells
    nb_path = os.path.join(NOTEBOOKS_DIR, "02_training_evaluation.ipynb")
    
    print(f"Executing and saving {nb_path}...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={'metadata': {'path': PROJECT_ROOT}})
    client.execute()
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("02_training_evaluation.ipynb successfully generated and executed with saved outputs!")


if __name__ == "__main__":
    build_eda_notebook()
    build_training_notebook()
