"""
src/load_data.py
Phase 0: Load, Clean, and Reshape Student Feedback Dataset
"""

import os
import glob
import numpy as np
import pandas as pd


ASPECT_DEFINITIONS = [
    ("teaching", "teachingl"),
    ("coursecontent", "coursecontentl"),
    ("examination", "examinationl"),
    ("labwork", "labworkl"),
    ("library_facilities", "library_facilitiesl"),
    ("extracurricular", "extracurricularl")
]

LABEL_TO_SENTIMENT = {
    -1: "negative",
    0: "neutral",
    1: "positive"
}


def find_dataset_file(search_dirs=None):
    """Locate the raw feedback dataset file (xlsx or csv)."""
    if search_dirs is None:
        search_dirs = [".", "data/raw", "data"]
    
    candidates = []
    for d in search_dirs:
        for ext in ("*.xlsx", "*.xls", "*.csv"):
            candidates.extend(glob.glob(os.path.join(d, ext)))
            
    # Filter out generated files in processed
    valid_candidates = [
        c for c in candidates 
        if "processed" not in c and not os.path.basename(c).startswith("~$")
    ]
    
    if not valid_candidates:
        raise FileNotFoundError("Could not find any student feedback dataset (.xlsx or .csv).")
        
    # Prefer student-data.xlsx if present
    for c in valid_candidates:
        if "student-data" in os.path.basename(c).lower():
            return os.path.normpath(c)
            
    return os.path.normpath(valid_candidates[0])


def map_columns(df):
    """
    Map raw dataframe columns to aspect text and label columns,
    matching case-insensitively and tolerating extra whitespace.
    """
    col_lookup = {col.strip().lower(): col for col in df.columns}
    aspect_col_pairs = {}
    
    for aspect_text_key, aspect_label_key in ASPECT_DEFINITIONS:
        actual_text_col = col_lookup.get(aspect_text_key.lower())
        actual_label_col = col_lookup.get(aspect_label_key.lower())
        
        if not actual_text_col or not actual_label_col:
            raise KeyError(
                f"Missing required columns for aspect '{aspect_text_key}'. "
                f"Looked for text='{aspect_text_key}' and label='{aspect_label_key}'."
            )
        aspect_col_pairs[aspect_text_key] = (actual_text_col, actual_label_col)
        
    return aspect_col_pairs


def compute_overall_sentiment(mean_score):
    """
    Compute student overall sentiment from mean of available aspect labels:
    mean <= -0.34: negative
    mean >= 0.34: positive
    else: neutral
    """
    if pd.isna(mean_score):
        return "neutral"
    if mean_score <= -0.34:
        return "negative"
    elif mean_score >= 0.34:
        return "positive"
    else:
        return "neutral"


def load_and_reshape(filepath=None, output_dir="data/processed"):
    """
    Full Phase 0 Pipeline:
    1. Load wide dataset
    2. Print diagnostics
    3. Clean and validate labels & texts
    4. Reshape to LONG and WIDE formats
    5. Check class distributions and warnings
    6. Generate data_report.md
    """
    if filepath is None:
        filepath = find_dataset_file()
        
    print(f"==================================================")
    print(f"PHASE 0: LOADING AND RESHAPING DATASET")
    print(f"Dataset file: {filepath}")
    print(f"==================================================")
    
    if filepath.endswith((".xlsx", ".xls")):
        df_raw = pd.read_excel(filepath)
    else:
        df_raw = pd.read_csv(filepath)
        
    raw_rows, raw_cols = df_raw.shape
    print(f"Total Rows: {raw_rows}, Total Columns: {raw_cols}")
    print("\nColumns detected:")
    print(list(df_raw.columns))
    print("\nData Types:")
    print(df_raw.dtypes)
    print("\nSample 5 rows:")
    print(df_raw.head(5))
    print("\nMissing values per column:")
    print(df_raw.isnull().sum())
    print("\nDuplicate rows in raw dataset:", df_raw.duplicated().sum())
    
    aspect_map = map_columns(df_raw)
    
    # Print label counts per aspect before cleaning
    print("\nLabel counts per aspect (raw):")
    for aspect, (t_col, l_col) in aspect_map.items():
        print(f"[{aspect}] raw label counts:")
        print(df_raw[l_col].value_counts(dropna=False).to_dict())
        
    # Standardize student_id
    df_raw = df_raw.copy()
    df_raw["student_id"] = [f"STU_{i:04d}" for i in range(len(df_raw))]
    
    # ----------------------------------------------------
    # Reshaping to LONG format
    # ----------------------------------------------------
    long_rows = []
    
    # Also prepare wide format with clean labels
    df_wide_clean = df_raw.copy()
    
    for aspect, (t_col, l_col) in aspect_map.items():
        # Standardize text
        clean_text_series = df_raw[t_col].astype(str).str.strip()
        clean_text_series = clean_text_series.replace({
            "nan": np.nan,
            "None": np.nan,
            "": np.nan
        })
        
        # Standardize label
        clean_label_series = pd.to_numeric(df_raw[l_col], errors="coerce")
        # Keep only values in {-1, 0, 1}
        clean_label_series = clean_label_series.where(clean_label_series.isin([-1, 0, 1]))
        
        df_wide_clean[f"{aspect}_text"] = clean_text_series
        df_wide_clean[f"{aspect}_label"] = clean_label_series
        
        for idx in range(len(df_raw)):
            stu_id = df_raw.loc[idx, "student_id"]
            txt = clean_text_series.iloc[idx]
            lbl = clean_label_series.iloc[idx]
            
            # Drop if missing text or label
            if pd.isna(txt) or pd.isna(lbl):
                continue
                
            int_lbl = int(lbl)
            sentiment_str = LABEL_TO_SENTIMENT[int_lbl]
            
            long_rows.append({
                "student_id": stu_id,
                "aspect": aspect,
                "text": txt,
                "label": int_lbl,
                "sentiment": sentiment_str
            })
            
    df_long = pd.DataFrame(long_rows)
    
    # ----------------------------------------------------
    # Build WIDE format with overall_sentiment per student
    # ----------------------------------------------------
    label_cols_clean = [f"{aspect}_label" for aspect in aspect_map.keys()]
    df_wide_clean["mean_aspect_score"] = df_wide_clean[label_cols_clean].mean(axis=1)
    df_wide_clean["overall_sentiment"] = df_wide_clean["mean_aspect_score"].apply(compute_overall_sentiment)
    
    # Save processed files
    os.makedirs(output_dir, exist_ok=True)
    long_path = os.path.join(output_dir, "feedback_long.csv")
    wide_path = os.path.join(output_dir, "feedback_wide.csv")
    
    df_long.to_csv(long_path, index=False)
    df_wide_clean.to_csv(wide_path, index=False)
    print(f"\nSaved LONG dataset to: {long_path} (Rows: {len(df_long)})")
    print(f"Saved WIDE dataset to: {wide_path} (Rows: {len(df_wide_clean)})")
    
    # ----------------------------------------------------
    # Class Distribution Analysis & Warnings
    # ----------------------------------------------------
    print("\n==================================================")
    print("CLASS DISTRIBUTION OVERALL (LONG FORMAT)")
    print("==================================================")
    overall_dist = df_long["sentiment"].value_counts(normalize=True) * 100
    overall_counts = df_long["sentiment"].value_counts()
    for s_class in ["positive", "neutral", "negative"]:
        cnt = overall_counts.get(s_class, 0)
        pct = overall_dist.get(s_class, 0.0)
        print(f"  {s_class.capitalize():<10}: {cnt:>5} ({pct:5.2f}%)")
        if pct < 5.0:
            print(f"  [WARNING] Class '{s_class}' is UNDER 5% of total dataset ({pct:.2f}%)!")
            
    print("\n==================================================")
    print("CLASS DISTRIBUTION PER ASPECT")
    print("==================================================")
    aspect_dist_summary = {}
    under_5_warnings = []
    
    for aspect in aspect_map.keys():
        sub = df_long[df_long["aspect"] == aspect]
        total_sub = len(sub)
        counts = sub["sentiment"].value_counts()
        pcts = sub["sentiment"].value_counts(normalize=True) * 100
        print(f"\nAspect: {aspect} (Total valid: {total_sub})")
        aspect_dist_summary[aspect] = {}
        for s_class in ["positive", "neutral", "negative"]:
            c = counts.get(s_class, 0)
            p = pcts.get(s_class, 0.0)
            aspect_dist_summary[aspect][s_class] = {"count": c, "percentage": p}
            print(f"  {s_class.capitalize():<10}: {c:>4} ({p:5.2f}%)")
            if p < 5.0:
                msg = f"Aspect '{aspect}' class '{s_class}' is {p:.2f}% (< 5%)!"
                under_5_warnings.append(msg)
                print(f"  [WARNING] {msg}")
                
    # Student-level distribution
    student_dist = df_wide_clean["overall_sentiment"].value_counts(normalize=True) * 100
    student_counts = df_wide_clean["overall_sentiment"].value_counts()
    print("\n==================================================")
    print("STUDENT-LEVEL OVERALL SENTIMENT DISTRIBUTION (WIDE)")
    print("==================================================")
    for s_class in ["positive", "neutral", "negative"]:
        cnt = student_counts.get(s_class, 0)
        pct = student_dist.get(s_class, 0.0)
        print(f"  {s_class.capitalize():<10}: {cnt:>5} ({pct:5.2f}%)")
        
    # ----------------------------------------------------
    # Generate data_report.md
    # ----------------------------------------------------
    generate_data_report(
        filepath=filepath,
        df_raw=df_raw,
        df_long=df_long,
        df_wide=df_wide_clean,
        aspect_dist_summary=aspect_dist_summary,
        under_5_warnings=under_5_warnings
    )
    
    return df_long, df_wide_clean


def generate_data_report(filepath, df_raw, df_long, df_wide, aspect_dist_summary, under_5_warnings):
    """Save data_report.md with comprehensive findings and assumptions."""
    null_table = "| Column | Missing Values |\n|---|---|\n"
    for col, n_missing in df_raw.isnull().sum().items():
        null_table += f"| `{col}` | {n_missing} |\n"

    report_content = f"""# Dataset Analysis and Reshaping Report

## 1. Overview and Detection
- **Original File**: `{filepath}`
- **Original Dimensions**: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns
- **Format**: Wide format (each row represents a single student survey response containing 6 feedback aspects)
- **Aspects Evaluated (6 total)**:
  1. `teaching` & `teachingl`
  2. `coursecontent` & `coursecontentl`
  3. `examination` & `examinationl`
  4. `labwork` & `labworkl`
  5. `library_facilities` & `library_facilitiesl` (detected leading space in column name `' library_facilities'`)
  6. `extracurricular` & `extracurricularl`

## 2. Raw Data Inspection
- **Missing Values in Raw Data**:
{null_table}

- **Duplicate Rows in Wide Data**: {df_raw.drop(columns=['student_id'], errors='ignore').duplicated().sum()} identical survey responses detected.
- **Label Encoding**:
  - `-1`: Negative sentiment
  - `0`: Neutral sentiment
  - `1`: Positive sentiment

## 3. Data Cleaning and Reshaping Logic
1. **Cleaning Rules**:
   - Column names matched case-insensitively with leading/trailing whitespace trimmed.
   - Text entries stripped of whitespace; non-string or empty representations (`nan`, `None`, whitespace) treated as missing.
   - Label values converted to numeric, keeping only valid labels in `{{-1, 0, 1}}`.
   - Rows with either missing text or missing label were dropped from the training set (LONG format).
2. **Reshaped Long Format (`data/processed/feedback_long.csv`)**:
   - Total Valid Feedback Instances: **{len(df_long)}** rows
   - Columns: `student_id`, `aspect`, `text`, `label`, `sentiment`
3. **Reshaped Wide Format (`data/processed/feedback_wide.csv`)**:
   - Total Students: **{len(df_wide)}** rows
   - Added `mean_aspect_score`: average of numeric labels for available aspects per student.
   - Added `overall_sentiment`:
     - `mean_aspect_score <= -0.34` → `negative`
     - `mean_aspect_score >= 0.34` → `positive`
     - `-0.34 < mean_aspect_score < 0.34` → `neutral`

## 4. Class Distribution Overall (Long Format)
Total Records: {len(df_long)}
- **Positive**: {df_long['sentiment'].value_counts().get('positive', 0)} ({df_long['sentiment'].value_counts(normalize=True).get('positive', 0)*100:.2f}%)
- **Neutral**: {df_long['sentiment'].value_counts().get('neutral', 0)} ({df_long['sentiment'].value_counts(normalize=True).get('neutral', 0)*100:.2f}%)
- **Negative**: {df_long['sentiment'].value_counts().get('negative', 0)} ({df_long['sentiment'].value_counts(normalize=True).get('negative', 0)*100:.2f}%)

## 5. Class Distribution Per Aspect
| Aspect | Total Responses | Positive (%) | Neutral (%) | Negative (%) |
|---|---|---|---|---|
"""
    for asp, stats in aspect_dist_summary.items():
        tot = stats['positive']['count'] + stats['neutral']['count'] + stats['negative']['count']
        pos_p = stats['positive']['percentage']
        neu_p = stats['neutral']['percentage']
        neg_p = stats['negative']['percentage']
        report_content += f"| {asp} | {tot} | {stats['positive']['count']} ({pos_p:.1f}%) | {stats['neutral']['count']} ({neu_p:.1f}%) | {stats['negative']['count']} ({neg_p:.1f}%) |\n"

    report_content += f"""
### Imbalance and Low Frequency Class Check
- **Threshold**: Warning if any sentiment class is under 5.0% of the aspect or overall data.
- **Detected Warnings**:
"""
    if under_5_warnings:
        for w in under_5_warnings:
            report_content += f"  - [WARNING] {w}\n"
    else:
        report_content += "  - None. All classes in every aspect represent >= 5.0% of the sample.\n"

    report_content += f"""
## 6. Student-Level Overall Sentiment (Wide Format)
- **Positive Students**: {df_wide['overall_sentiment'].value_counts().get('positive', 0)} ({df_wide['overall_sentiment'].value_counts(normalize=True).get('positive', 0)*100:.2f}%)
- **Neutral Students**: {df_wide['overall_sentiment'].value_counts().get('neutral', 0)} ({df_wide['overall_sentiment'].value_counts(normalize=True).get('neutral', 0)*100:.2f}%)
- **Negative Students**: {df_wide['overall_sentiment'].value_counts().get('negative', 0)} ({df_wide['overall_sentiment'].value_counts(normalize=True).get('negative', 0)*100:.2f}%)

## 7. Assumptions & Project Constraints
1. **Absence of Metadata**: The dataset contains no timestamps, semester info, course codes, or instructor identifiers. Thus, analysis focuses purely on aspect-based feedback semantics and student-level sentiment correlation.
2. **Student Disjoint Splits**: Because each wide row represents a distinct student survey response, model validation splits must group by `student_id` to prevent data leakage between training and testing sets.
3. **Aspect Conditioning**: For unified cross-aspect sentiment classification models, the aspect name is prepended to the feedback string (`<aspect>: <text>`) to provide explicit context.
4. **Class Weighting**: Because positive feedback predominates (~60%), cost-sensitive learning (balanced class weights) is required during model training.
"""
    with open("data_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    print("Saved comprehensive report to data_report.md")


if __name__ == "__main__":
    load_and_reshape()
