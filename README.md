# 🎓 Student Feedback Analysis using NLP
> **Deconstruct student voices into actionable academic insights with Aspect-Based Sentiment Analysis.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![HuggingFace DistilBERT](https://img.shields.io/badge/Transformers-DistilBERT-yellow?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

---

## 📌 Description

Universities gather thousands of open-ended survey reviews each term, but generic sentiment analysis yields only a single high-level polarity—masking critical details where a student might praise teaching quality while expressing severe frustration with laboratory equipment. 

**Student Feedback Analysis** is an end-to-end Natural Language Processing (NLP) pipeline and interactive web application that performs **Aspect-Based Sentiment Analysis (ABSA)** across six core pillars of campus life. It pinpoints positive, neutral, and negative sentiment per domain, extracts root negative themes, and equips academic administrators with data-backed decisions.

---

## 🖼️ Dashboard Preview

<div align="center">
  <img src="https://via.placeholder.com/1100x560/0f172a/38bdf8?text=Student+Feedback+Analysis+Dashboard+Preview+%7C+Streamlit+UI" alt="Student Feedback Analysis Dashboard Preview" width="100%" />
  <p><em>Interactive 5-page Streamlit Dashboard featuring live inference, batch analytics, and topic modeling.</em></p>
</div>

---

## ✨ Key Features

- **🎯 6-Aspect ABSA Decomposition**: Evaluates sentiment across *Teaching*, *Course Content*, *Examination*, *Lab Work*, *Library Facilities*, and *Extracurricular Activities*.
- **🧠 Negation-Aware Preprocessing**: Preserves crucial contextual phrases (e.g., *"not helpful"*, *"no practical exposure"*) via custom NLTK tokenization and WordNet lemmatization.
- **⚡ Multi-Model Benchmarking**: Empirically compares five distinct architectures:
  - **VADER** (Lexicon Rule Baseline)
  - **TF-IDF + Logistic Regression** *(Best Production Model - Macro F1: ~0.84)*
  - **TF-IDF + Multinomial Naive Bayes**
  - **TF-IDF + Calibrated Linear SVM**
  - **Fine-Tuned DistilBERT** (Deep contextual transformer)
- **📊 Interactive Streamlit Dashboard**:
  - **Batch Analysis**: Upload `.xlsx` / `.csv` survey sheets for immediate campus-wide diagnostics.
  - **Live Inference**: Test custom reviews conditioned on specific aspects in real-time.
  - **Root-Cause Discovery**: Topic modeling via Latent Dirichlet Allocation (LDA) to pinpoint negative student friction points.
  - **Cross-Aspect Heatmaps**: Pearson correlation maps showing interconnected satisfaction trends.

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Core & Data** | Python 3.10+, Pandas, NumPy, OpenPyXL |
| **NLP & Preprocessing** | NLTK (WordNet, Stopwords), VADER Sentiment |
| **Machine Learning** | Scikit-learn (TF-IDF, Logistic Regression, Linear SVM, Naive Bayes, LDA), Joblib |
| **Deep Learning** | Hugging Face Transformers, PyTorch (DistilBERT) |
| **Visualization** | Plotly Express & Graph Objects, Seaborn, Matplotlib, WordCloud |
| **Web Interface** | Streamlit |

---

## 🚀 5-Minute Quickstart

Follow these simple steps to set up and run the application locally:

### 1. Clone the Repository
```bash
git clone https://github.com/Sahilattar/Student-Feedback-Analysis-ABSA-project.git
cd Student-Feedback-Analysis-ABSA-project
```

### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Dashboard
Run the Streamlit application:
```bash
streamlit run app.py
```
*(On Windows, you can also just double-click `run_app.bat`)*

The dashboard will automatically open in your browser at:
```
http://localhost:8501
```

---

## 📂 Project Directory Structure

```text
Student_Feedback_Analysis/
├── app.py                     # Streamlit web dashboard entry point
├── run_app.bat                # Windows one-click launcher
├── requirements.txt           # Python package dependencies
├── student-data.xlsx          # Raw student survey dataset (222 students)
├── src/
│   ├── load_data.py           # Ingestion and dual-format reshaping (Wide & Long)
│   ├── preprocess.py          # Negation-preserving text normalization
│   ├── train.py               # ML benchmarking & serialization
│   ├── predict.py             # Inference engine (aspect-conditioned)
│   └── analysis.py            # LDA topic modeling & aspect correlation heatmaps
├── models/                    # Saved model artifacts (.joblib & DistilBERT weights)
└── data/                      # Processed datasets and stratified splits
```

---

## 🌐 Deployment to Render

To deploy this application to **[Render](https://render.com/)**:
1. Connect your GitHub repository to a new **Web Service**.
2. Set **Runtime** to `Python 3`.
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**:
   ```bash
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```

---

## 👤 Author

**Sahil Attar**
- GitHub: [@Sahilattar](https://github.com/Sahilattar)
- Project: [Student-Feedback-Analysis-ABSA-project](https://github.com/Sahilattar/Student-Feedback-Analysis-ABSA-project)

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
