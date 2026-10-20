# Project Explanation: Student Feedback Analysis using NLP
## How to Explain This Project from Memory in an Interview or Viva

This document breaks down every folder, file, and function in simple English. Read this before your viva or project presentation so you can confidently explain the entire codebase without looking at the code.

---

## 1. High-Level Summary (The 30-Second Elevator Pitch)
"Our project builds an Aspect-Based Sentiment Analysis system for university feedback. When students fill out term surveys, their opinions cover different areas: teaching, course content, exams, labs, library, and extracurricular activities. Instead of giving one vague sentiment score for the entire college, our NLP pipeline breaks comments down into each aspect, normalizes the text while keeping critical negation words like 'not' and 'never', and trains five different models using a student-disjoint split to avoid data leakage. Our best model, TF-IDF with balanced Logistic Regression, achieves a 70.97% Macro F1 score on unseen students. We also uncover negative complaint themes using topic modeling, prove that teaching quality most strongly drives overall student satisfaction, and deliver the system via an interactive Streamlit web dashboard."

---

## 2. Directory Structure Overview

- **`data/`**:
  - `data/raw/student-data.xlsx`: Original student survey spreadsheet.
  - `data/processed/feedback_long.csv`: Cleaned dataset in long format (one row per student-aspect feedback).
  - `data/processed/feedback_wide.csv`: Cleaned dataset in wide format with calculated `overall_sentiment` per student.
- **`src/`**:
  - `src/load_data.py`: Loads the raw data, cleans labels, and reshapes into long and wide formats.
  - `src/preprocess.py`: Cleans raw text, handles negation contractions, removes stopwords, and lemmatizes.
  - `src/train.py`: Splits data by student, trains 5 models, evaluates them on the test set, and saves the best model.
  - `src/predict.py`: Provides inference functions for single comments and batch spreadsheets.
  - `src/analysis.py`: Performs aspect ranking, negative topic modeling, word cloud generation, and correlation analysis.
- **`models/`**:
  - `best_model.joblib`: Serialized best classifier (Logistic Regression).
  - `tfidf_vectorizer.joblib`: Fitted TF-IDF feature extractor.
  - `label_encoder.joblib`: Class label encoder.
  - `model_metadata.json`: Pinned test metrics, per-aspect F1 scores, and configuration.
  - `distilbert_feedback/`: Fine-tuned transformer weights and tokenizer.
- **`reports/`**:
  - Confusion matrix plots (`cm_*.png`), model comparison chart and CSV, word clouds, correlation heatmaps, and key insights report.
- **`notebooks/`**:
  - `01_eda.ipynb`: Step-by-step exploratory data analysis with executed charts.
  - `02_training_evaluation.ipynb`: Complete ML training, evaluation, and inference pipeline with saved outputs.
- **`app.py`**: Streamlit interactive web application with 5 sidebar pages.
- **`requirements.txt`**: Pinned library dependencies.
- **`README.md`**: Master project documentation with results and 15 viva Q&As.

---

## 3. Deep-Dive into Every File and Function

### A. `src/load_data.py` (Phase 0: Ingestion and Reshaping)
**Purpose**: Reads the messy raw Excel survey and creates two clean, structured datasets.

1. **`find_dataset_file()`**:
   - *What it does*: Automatically scans the project folder and `data/raw/` for `.xlsx` or `.csv` files.
   - *Why*: Ensures the code runs seamlessly on any machine without hardcoded paths.
2. **`map_columns(df)`**:
   - *What it does*: Matches column names case-insensitively and tolerates extra spaces. For example, it detects `' library_facilities'` (which had a leading space in the raw Excel) and pairs it with `'library_facilitiesl'`.
3. **`compute_overall_sentiment(mean_score)`**:
   - *What it does*: Calculates a student's institutional satisfaction from their average aspect rating.
   - *Rule*: Mean $\le -0.34 \rightarrow$ Negative; Mean $\ge 0.34 \rightarrow$ Positive; Otherwise $\rightarrow$ Neutral.
4. **`load_and_reshape(filepath, output_dir)`**:
   - *What it does*: Converts the wide table (222 rows × 12 columns) into:
     - `feedback_long.csv` (1,324 rows: `student_id`, `aspect`, `text`, `label`, `sentiment`). Rows with missing text or invalid labels are dropped.
     - `feedback_wide.csv` (222 student rows with `overall_sentiment`).
   - Checks if any class represents under 5% of the data (none do!).
5. **`generate_data_report(...)`**:
   - *What it does*: Writes `data_report.md` recording dataset dimensions, null counts, duplicate rows, class distributions, and explicit assumptions.

---

### B. `src/preprocess.py` (Phase 2: Text Preprocessing)
**Purpose**: Transforms noisy student comments into normalized text tokens ready for machine learning.

1. **`clean_text(text)`**:
   - *What it does*:
     1. Converts text to lowercase.
     2. Removes HTML tags, web URLs, and email addresses with regular expressions.
     3. **Expands negation contractions** before stripping punctuation: turns `don't` into `do not`, `can't` into `can not`, and `won't` into `will not`.
     4. Removes digits and punctuation.
     5. Tokenizes words and filters English stopwords **while strictly preserving negation words**: `not`, `no`, `never`, `nor`, and `n't`.
     6. Lemmatizes each remaining token using NLTK's `WordNetLemmatizer` (e.g. `explaining` $\rightarrow$ `explain`, `computers` $\rightarrow$ `computer`).
     7. Strips and collapses whitespace into a single clean string.
   - *Why this matters*: If you drop the word "not", the phrase *"not good"* becomes *"good"*, completely corrupting your sentiment classifier!
2. **`preprocess_dataset(df, text_col, label_col, drop_duplicates)`**:
   - *What it does*: Applies `clean_text` across a DataFrame, drops empty cleaned strings, and removes exact duplicate feedback entries.

---

### C. `src/train.py` (Phase 3: Model Training and Evaluation)
**Purpose**: Trains, benchmarks, and evaluates five different sentiment classifiers and saves the winning model.

1. **`get_data_splits()`**:
   - *What it does*: Creates a **student-disjoint 70/15/15 train/val/test split**.
   - *How*: It groups by `student_id` and stratifies on `overall_sentiment`. This ensures 155 students are in Train (927 comments), 33 students in Val (198 comments), and 34 students in Test (199 comments).
   - *Crucial Concept*: **Zero student overlap**. If Student #5 writes comments for teaching and lab work, both comments stay together in the same split. They never leak into the test set!
   - Prepend aspect name: Creates `feature_text = aspect + ": " + clean_text` (e.g. `"teaching: teacher clear explanation"`).
2. **`train_vader_baseline(test_df)`**:
   - *What it does*: Evaluates the off-the-shelf VADER rule-based sentiment dictionary on the test set.
   - *Rule*: Compound score $\ge 0.05 \rightarrow$ Positive, $\le -0.05 \rightarrow$ Negative, otherwise Neutral.
3. **`train_traditional_models(train_df, test_df)`**:
   - *What it does*: Fits a `TfidfVectorizer(ngram_range=(1,2), min_df=2)` and trains three classical models:
     - **Logistic Regression**: Uses balanced class weights (`class_weight='balanced'`) to penalize errors on minority classes.
     - **Multinomial Naive Bayes**: Fast probabilistic classifier based on Bayes' theorem.
     - **Linear SVM (Calibrated)**: Wraps `LinearSVC` with `CalibratedClassifierCV` (Platt scaling) to output calibrated probabilities for confidence scoring.
4. **`train_distilbert_model(train_df, val_df, test_df)`**:
   - *What it does*: Fine-tunes `distilbert-base-uncased` (66M parameter transformer) for 2 epochs on the training set using CrossEntropyLoss with class weights and AdamW optimizer.
5. **`evaluate_predictions(y_true, y_pred, model_name)`**:
   - *What it does*: Computes Accuracy, Macro Precision, Macro Recall, and Macro F1, and plots a normalized confusion matrix heatmap (`reports/cm_*.png`).
6. **`plot_model_comparison(df_comp)`**:
   - *What it does*: Saves a comparative bar chart (`reports/model_comparison.png`).
7. **`compute_per_aspect_f1(test_df, best_preds)`**:
   - *What it does*: Breaks down the Macro F1 and Accuracy of the best model across each of the 6 individual aspects.
8. **`analyze_misclassifications(test_df, best_preds, best_model_name)`**:
   - *What it does*: Finds all misclassified test cases, extracts 10 diverse examples, and categorizes reasons (brevity, mixed sentiment, neutral boundary ambiguity, sarcasm). Saves to `reports/misclassified_examples.md`.
9. **`run_training_pipeline()`**:
   - *What it does*: Coordinates the entire training workflow, picks the highest Macro F1 model (Logistic Regression, 0.7097), and serializes artifacts to `models/`.

---

### D. `src/predict.py` (Inference Engine)
**Purpose**: Production inference module for predicting single comments or batch files.

1. **`SentimentPredictor` class**:
   - Loads `best_model.joblib`, `tfidf_vectorizer.joblib`, and `label_encoder.joblib`.
2. **`predict_sentiment(text, aspect)`**:
   - Preprocesses input text, transforms it using the TF-IDF vectorizer, and queries the model.
   - Returns a structured dictionary:
     - `sentiment`: `"positive"`, `"neutral"`, or `"negative"`.
     - `confidence`: Highest probability score (0.0 to 1.0).
     - `probabilities`: Dictionary of individual class probabilities.
     - `clean_text`: Normalized string.
3. **`predict_dataframe(df, text_col, aspect_col)`**:
   - Iterates through a pandas DataFrame and adds prediction and confidence columns.
4. **Command-Line Interface (CLI)**:
   - Allows running `python src/predict.py --text "Great professor" --aspect "teaching"` directly from terminal.

---

### E. `src/analysis.py` (Phase 4: Extra Analytics & Topic Modeling)
**Purpose**: Generates high-level institutional insights, topic modeling, word clouds, and correlations.

1. **`analyze_aspect_sentiment(df_long)`**:
   - Calculates percentage of positive, neutral, and negative comments per aspect.
   - Computes **Net Sentiment Score** (`% Positive - % Negative`) and ranks aspects:
     1. Extracurricular (+53.2%)
     2. Teaching (+38.7%)
     3. Examination (+38.4%)
     4. Course Content (+38.0%)
     5. Library Facilities (+33.8%)
     6. Lab Work (+32.1%)
   - Saves stacked bar chart `reports/aspect_sentiment_stacked.png`.
2. **`perform_topic_modeling_negative(df_long)`**:
   - Filters data to negative comments only.
   - Fits **Latent Dirichlet Allocation (LDA)** to extract 3 distinct topics per aspect.
   - Uses keyword mapping heuristics to assign clear descriptive labels (e.g. *"Malfunctioning Hardware & Slow Workstations"* for labs, *"Shortage of Latest Textbook Editions"* for library).
   - Saves results to `reports/negative_topics.json`.
3. **`generate_wordclouds(df_long)`**:
   - Uses the `WordCloud` library to generate:
     - Overall positive and negative word clouds.
     - 12 aspect-specific word clouds (positive and negative for each aspect) saved in `reports/wordclouds/`.
4. **`analyze_student_correlations(df_wide)`**:
   - Computes Pearson correlation matrix between all numeric aspect labels and student overall sentiment.
   - **Key Finding**: `teaching` has the highest correlation ($r = 0.7609$), meaning teaching quality is the primary driver of student institutional satisfaction.
   - Saves `reports/aspect_correlation_heatmap.png`.
5. **`analyze_comment_length(df_long)`**:
   - Measures word and character lengths by sentiment.
   - **Key Finding**: Negative comments average 11.8 words (70.2 characters), while positive comments average only 4.5 words (27.1 characters). Dissatisfied students write nearly 3x more text.
   - Saves `reports/comment_length_by_sentiment.png`.
6. **`generate_key_insights_report(...)`**:
   - Generates `reports/key_insights.md` summarizing 8 empirical findings and 5 concrete recommendations for university deans.

---

### F. `app.py` (Phase 5: Streamlit Web Dashboard)
**Purpose**: User-friendly web interface for student feedback analysis.

1. **Page 1: 📤 Upload & Analyze**:
   - Supports dragging and dropping `.csv` or `.xlsx` survey files.
   - Automatically detects wide aspect columns vs single-text columns.
   - Generates aspect-wise stacked charts, sentiment pie charts, interactive tables, and a CSV download button.
2. **Page 2: 💡 Insights & Findings**:
   - Visualizes aspect rankings, negative LDA topics, high-res word clouds, and the correlation heatmap.
3. **Page 3: 🧪 Try It (Live Predictor)**:
   - Interactive sandbox where users select an aspect, type or paste any comment, and view live predicted sentiment, confidence percentage, probability distribution bars, and preprocessed tokens. Includes 4 one-click preset buttons.
4. **Page 4: 📈 Model Performance**:
   - Shows benchmark tables, comparison bar chart, confusion matrix inspector dropdown, per-aspect F1 scores, and plain-English architectural explanations.
5. **Page 5: ℹ️ About & Methodology**:
   - Outlines problem statement, dataset details, ASCII pipeline diagram, and project limitations.

---

## 4. Key Metrics to Memorize for Your Presentation

- **Total Feedback Instances**: 1,324 aspect comments from 222 students.
- **Overall Class Split**: Positive ~61%, Negative ~22%, Neutral ~18%.
- **Train/Val/Test Split**: 70% Train (155 students / 927 comments), 15% Val (33 students / 198 comments), 15% Test (34 students / 199 comments).
- **Best Model**: TF-IDF + Logistic Regression (Balanced).
  - **Macro F1**: **0.7097** (70.97%)
  - **Test Accuracy**: **77.89%**
  - **Macro Recall**: **72.41%**
- **DistilBERT Macro F1**: **0.7044** (Macro Recall: 73.05%).
- **Linear SVM Accuracy**: **79.90%** (Macro F1: 0.6719).
- **VADER Baseline Macro F1**: **0.6110** (Accuracy: 76.38%).
- **Top Driver of Overall Satisfaction**: Teaching ($r = 0.7609$).
- **Worst Rated Aspect**: Lab Work (26.7% negative feedback).
- **Best Rated Aspect**: Extracurricular (69.4% positive feedback).
