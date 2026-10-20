"""
app.py
Streamlit Web Application: Student Feedback Analysis using NLP
Interactive dashboard for Aspect-Based Sentiment Analysis.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import clean_text
from src.predict import get_predictor
from src.load_data import ASPECT_DEFINITIONS, compute_overall_sentiment

# Set page configuration
st.set_page_config(
    page_title="Student Feedback Analysis (NLP)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling and sentiment badges
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .badge-positive {
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-neutral {
        background-color: #f3f4f6;
        color: #374151;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-negative {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Helper function for sentiment badges
def sentiment_badge(sentiment: str) -> str:
    s = sentiment.lower()
    if s == "positive":
        return '<span class="badge-positive">POSITIVE</span>'
    elif s == "negative":
        return '<span class="badge-negative">NEGATIVE</span>'
    else:
        return '<span class="badge-neutral">NEUTRAL</span>'

# Color constants
COLOR_MAP = {
    "positive": "#10b981", # Emerald green
    "neutral": "#9ca3af",  # Gray
    "negative": "#ef4444"  # Red
}

# Sidebar navigation
st.sidebar.image("https://img.icons8.com/clouds/200/graduation-cap.png", width=110)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "📤 Upload & Analyze",
        "💡 Insights & Findings",
        "🧪 Try It (Live Predictor)",
        "📈 Model Performance",
        "ℹ️ About & Methodology"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project**: Aspect-Based Sentiment Analysis")
st.sidebar.markdown("**Models**: TF-IDF, SVM, VADER, DistilBERT")
st.sidebar.markdown("**Dataset**: 6 Core Academic Aspects")


# ==============================================================================
# PAGE 1: UPLOAD & ANALYZE
# ==============================================================================
if page == "📤 Upload & Analyze":
    st.markdown('<div class="main-title">📤 Upload & Batch Feedback Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload a student feedback spreadsheet (CSV or XLSX) to analyze sentiment per aspect and evaluate student satisfaction.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a CSV or XLSX file",
        type=["csv", "xlsx", "xls"],
        help="Upload feedback surveys in wide format (aspect columns) or single text column format."
    )

    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith((".xlsx", ".xls")):
                df_upload = pd.read_excel(uploaded_file)
            else:
                df_upload = pd.read_csv(uploaded_file)

            if df_upload.empty:
                st.error("Uploaded file is empty. Please upload a valid dataset.")
                st.stop()

            st.success(f"Successfully loaded `{uploaded_file.name}` with {len(df_upload)} rows and {len(df_upload.columns)} columns.")

            # Check if dataset has standard wide aspect columns
            col_lookup = {c.strip().lower(): c for c in df_upload.columns}
            aspect_matches = {}
            for asp, _ in ASPECT_DEFINITIONS:
                if asp.lower() in col_lookup:
                    aspect_matches[asp] = col_lookup[asp.lower()]

            predictor = get_predictor()

            # SCENARIO A: Wide aspect-based dataset
            if len(aspect_matches) >= 3:
                st.info(f"Detected **Wide Multi-Aspect Survey Dataset** with {len(aspect_matches)} feedback aspect columns.")
                
                with st.spinner("Analyzing feedback across all aspects using best model..."):
                    long_records = []
                    for idx, row in df_upload.iterrows():
                        stu_id = f"Student_{idx+1}"
                        for asp, actual_col in aspect_matches.items():
                            val = row[actual_col]
                            if pd.notna(val) and str(val).strip():
                                pred = predictor.predict_sentiment(str(val), aspect=asp)
                                long_records.append({
                                    "student_id": stu_id,
                                    "aspect": asp,
                                    "original_text": str(val),
                                    "clean_text": pred["clean_text"],
                                    "predicted_sentiment": pred["sentiment"],
                                    "confidence": pred["confidence"],
                                    "prob_positive": pred["probabilities"].get("positive", 0.0),
                                    "prob_neutral": pred["probabilities"].get("neutral", 0.0),
                                    "prob_negative": pred["probabilities"].get("negative", 0.0)
                                })
                    res_df = pd.DataFrame(long_records)

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                total_comments = len(res_df)
                pos_pct = (res_df["predicted_sentiment"] == "positive").mean() * 100
                neu_pct = (res_df["predicted_sentiment"] == "neutral").mean() * 100
                neg_pct = (res_df["predicted_sentiment"] == "negative").mean() * 100

                col1.metric("Total Responses", f"{total_comments:,}")
                col2.metric("Positive", f"{pos_pct:.1f}%")
                col3.metric("Neutral", f"{neu_pct:.1f}%")
                col4.metric("Negative", f"{neg_pct:.1f}%")

                st.markdown("### Aspect-Wise Sentiment Breakdown")
                aspect_ct = pd.crosstab(res_df["aspect"], res_df["predicted_sentiment"], normalize="index") * 100
                for col in ["positive", "neutral", "negative"]:
                    if col not in aspect_ct.columns:
                        aspect_ct[col] = 0.0
                aspect_ct = aspect_ct[["positive", "neutral", "negative"]].reset_index()

                fig_stacked = px.bar(
                    aspect_ct,
                    y="aspect",
                    x=["positive", "neutral", "negative"],
                    orientation="h",
                    color_discrete_map=COLOR_MAP,
                    title="Predicted Sentiment Proportions per Aspect (%)",
                    labels={"value": "Percentage (%)", "aspect": "Academic Aspect", "variable": "Sentiment"}
                )
                fig_stacked.update_layout(barmode="stack", height=400, legend_title_text="Sentiment")
                st.plotly_chart(fig_stacked, use_container_width=True)

                st.markdown("### Detailed Predictions Table")
                st.dataframe(
                    res_df[["student_id", "aspect", "original_text", "predicted_sentiment", "confidence"]].head(100),
                    use_container_width=True
                )

                # Download results button
                csv_data = res_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Full Predictions as CSV",
                    data=csv_data,
                    file_name="feedback_predictions.csv",
                    mime="text/csv",
                    key="download_wide"
                )

            # SCENARIO B: Single text column dataset
            else:
                st.info("Single text column format detected. Select the column containing student feedback:")
                text_col = st.selectbox("Text Column", df_upload.columns.tolist())
                aspect_choice = st.selectbox(
                    "Assign Academic Aspect (Optional context)",
                    ["teaching", "coursecontent", "examination", "labwork", "library_facilities", "extracurricular"]
                )

                if st.button("Run Sentiment Analysis", type="primary"):
                    with st.spinner("Analyzing comments..."):
                        preds = []
                        confs = []
                        clean_tx = []
                        for txt in df_upload[text_col]:
                            p = predictor.predict_sentiment(str(txt), aspect=aspect_choice)
                            clean_tx.append(p["clean_text"])
                            preds.append(p["sentiment"])
                            confs.append(p["confidence"])

                        res_df = df_upload.copy()
                        res_df["cleaned_text"] = clean_tx
                        res_df["predicted_sentiment"] = preds
                        res_df["confidence"] = confs

                    col1, col2, col3 = st.columns(3)
                    pos_pct = (res_df["predicted_sentiment"] == "positive").mean() * 100
                    neu_pct = (res_df["predicted_sentiment"] == "neutral").mean() * 100
                    neg_pct = (res_df["predicted_sentiment"] == "negative").mean() * 100

                    col1.metric("Positive Feedback", f"{pos_pct:.1f}%")
                    col2.metric("Neutral Feedback", f"{neu_pct:.1f}%")
                    col3.metric("Negative Feedback", f"{neg_pct:.1f}%")

                    # Pie chart
                    fig_pie = px.pie(
                        res_df,
                        names="predicted_sentiment",
                        color="predicted_sentiment",
                        color_discrete_map=COLOR_MAP,
                        title=f"Sentiment Distribution for '{text_col}'"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                    st.dataframe(res_df.head(50), use_container_width=True)

                    csv_data = res_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_data,
                        file_name="predictions_output.csv",
                        mime="text/csv",
                        key="download_single"
                    )

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("Please ensure your uploaded spreadsheet contains valid feedback strings.")


# ==============================================================================
# PAGE 2: INSIGHTS & FINDINGS
# ==============================================================================
elif page == "💡 Insights & Findings":
    st.markdown('<div class="main-title">💡 Deep Analytical Insights & Findings</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Institutional analytics, aspect rankings, negative topic extraction, and correlation analysis.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Aspect Ranking",
        "🔍 Negative Topic Modeling",
        "☁️ Word Clouds",
        "📊 Student Correlations & Insights"
    ])

    with tab1:
        st.subheader("Academic Aspect Rankings (Best to Worst)")
        st.markdown("Ranked by **Net Sentiment Score** (`% Positive - % Negative`). Higher net score denotes superior student satisfaction.")
        
        aspect_img_path = os.path.join(PROJECT_ROOT, "reports", "aspect_sentiment_stacked.png")
        if os.path.exists(aspect_img_path):
            st.image(aspect_img_path, caption="Stacked Aspect Sentiment Proportions", use_container_width=True)
        else:
            st.warning("Chart image not found. Please run 'python src/analysis.py' first.")

        st.markdown("""
        **Summary Ranking**:
        1. 🥇 **Extracurricular Activities** (Net Score: **+53.2%**) - Vibrant sports and campus engagement.
        2. 🥈 **Teaching Quality** (Net Score: **+38.7%**) - Core instructional competence and faculty dedication.
        3. 🥉 **Examination System** (Net Score: **+38.4%**) - Standardized evaluation framework.
        4. **Course Content** (Net Score: **+38.0%**) - Comprehensive fundamental theory.
        5. **Library Facilities** (Net Score: **+33.8%**) - Physical catalog and study spaces.
        6. ⚠️ **Lab Work** (Net Score: **+32.1%**) - Highest negative proportion (26.7%); requires immediate lab modernization.
        """)

    with tab2:
        st.subheader("Negative Feedback Topic Modeling (LDA)")
        st.markdown("Latent Dirichlet Allocation (LDA) applied strictly to negative feedback per aspect to isolate specific student pain points:")

        topics_json_path = os.path.join(PROJECT_ROOT, "reports", "negative_topics.json")
        if os.path.exists(topics_json_path):
            with open(topics_json_path, "r", encoding="utf-8") as f:
                topics_data = json.load(f)

            cols = st.columns(2)
            idx = 0
            for aspect, t_list in topics_data.items():
                with cols[idx % 2]:
                    with st.expander(f"📌 {aspect.replace('_', ' ').title()} - Negative Themes", expanded=True):
                        for t in t_list:
                            st.markdown(f"**Topic {t['topic_id']}: {t['label']}**")
                            st.caption(f"Keywords: `{', '.join(t['keywords'])}`")
                idx += 1
        else:
            st.warning("Topics JSON not found. Please run 'python src/analysis.py'.")

    with tab3:
        st.subheader("High-Resolution Word Clouds")
        wc_choice = st.selectbox(
            "Select Word Cloud View",
            [
                "Overall Positive Comments",
                "Overall Negative Comments",
                "Teaching (Positive vs Negative)",
                "Lab Work (Positive vs Negative)",
                "Library Facilities (Positive vs Negative)",
                "Course Content (Positive vs Negative)",
                "Examination (Positive vs Negative)",
                "Extracurricular (Positive vs Negative)"
            ]
        )

        if wc_choice == "Overall Positive Comments":
            p = os.path.join(PROJECT_ROOT, "reports", "wordcloud_positive_overall.png")
            if os.path.exists(p):
                st.image(p, caption="Most Frequent Terms in Positive Student Feedback", use_container_width=True)
        elif wc_choice == "Overall Negative Comments":
            p = os.path.join(PROJECT_ROOT, "reports", "wordcloud_negative_overall.png")
            if os.path.exists(p):
                st.image(p, caption="Most Frequent Terms in Negative Student Feedback", use_container_width=True)
        else:
            asp_key = wc_choice.split(" ")[0].lower().replace("course", "coursecontent").replace("lab", "labwork").replace("library", "library_facilities")
            col_a, col_b = st.columns(2)
            pos_p = os.path.join(PROJECT_ROOT, "reports", "wordclouds", f"wc_{asp_key}_positive.png")
            neg_p = os.path.join(PROJECT_ROOT, "reports", "wordclouds", f"wc_{asp_key}_negative.png")
            with col_a:
                if os.path.exists(pos_p):
                    st.image(pos_p, caption=f"{asp_key.title()} - Positive", use_container_width=True)
            with col_b:
                if os.path.exists(neg_p):
                    st.image(neg_p, caption=f"{asp_key.title()} - Negative", use_container_width=True)

    with tab4:
        st.subheader("Student-Level Aspect Correlation & Key Findings")
        col_c, col_d = st.columns(2)
        with col_c:
            corr_path = os.path.join(PROJECT_ROOT, "reports", "aspect_correlation_heatmap.png")
            if os.path.exists(corr_path):
                st.image(corr_path, caption="Aspect Correlation with Overall Student Sentiment", use_container_width=True)
        with col_d:
            dist_path = os.path.join(PROJECT_ROOT, "reports", "student_overall_sentiment_dist.png")
            if os.path.exists(dist_path):
                st.image(dist_path, caption="Distribution of Student Overall Sentiment", use_container_width=True)

        st.markdown("### Institutional Action Items & Key Recommendations")
        insights_md_path = os.path.join(PROJECT_ROOT, "reports", "key_insights.md")
        if os.path.exists(insights_md_path):
            with open(insights_md_path, "r", encoding="utf-8") as f:
                md_text = f.read()
            st.markdown(md_text)


# ==============================================================================
# PAGE 3: TRY IT (LIVE PREDICTOR)
# ==============================================================================
elif page == "🧪 Try It (Live Predictor)":
    st.markdown('<div class="main-title">🧪 Try It: Real-Time Aspect Sentiment Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter any student comment, pick an academic aspect, and inspect real-time sentiment predictions and class probabilities.</div>', unsafe_allow_html=True)

    col_input, col_meta = st.columns([2, 1])

    with col_meta:
        aspect = st.selectbox(
            "Evaluation Aspect",
            [
                "teaching",
                "coursecontent",
                "examination",
                "labwork",
                "library_facilities",
                "extracurricular"
            ],
            format_func=lambda x: x.replace("_", " ").title()
        )

        st.markdown("**Quick Preset Examples:**")
        sample_choice = st.radio(
            "Choose a test sample:",
            [
                "Custom input",
                "Positive: 'The professor gives very intuitive real-world examples.'",
                "Negative: 'Lab computers are extremely slow and the software is outdated.'",
                "Neutral: 'The syllabus covers standard concepts as expected.'",
                "Negation: 'The teachers are not explaining concepts properly at all.'"
            ]
        )

    # Populate text area based on preset
    preset_text = ""
    if sample_choice.startswith("Positive:"):
        preset_text = "The professor gives very intuitive real-world examples."
    elif sample_choice.startswith("Negative:"):
        preset_text = "Lab computers are extremely slow and the software is outdated."
    elif sample_choice.startswith("Neutral:"):
        preset_text = "The syllabus covers standard concepts as expected."
    elif sample_choice.startswith("Negation:"):
        preset_text = "The teachers are not explaining concepts properly at all."

    with col_input:
        user_text = st.text_area(
            "Enter Student Feedback Comment:",
            value=preset_text,
            height=130,
            placeholder="Type or paste student comment here..."
        )
        predict_btn = st.button("🔮 Predict Sentiment", type="primary")

    if predict_btn or preset_text:
        if not user_text.strip():
            st.warning("Please type a comment before predicting.")
        else:
            predictor = get_predictor()
            res = predictor.predict_sentiment(user_text, aspect=aspect)

            st.markdown("---")
            st.subheader("Prediction Result")

            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.markdown(f"**Predicted Sentiment:**")
                st.markdown(sentiment_badge(res["sentiment"]), unsafe_allow_html=True)

            with res_col2:
                st.markdown(f"**Confidence Score:**")
                st.metric("Confidence", f"{res['confidence']*100:.1f}%")

            with res_col3:
                st.markdown(f"**Target Aspect:**")
                st.code(res["aspect"].replace("_", " ").title())

            # Probability Distribution Bar
            st.markdown("##### Class Probability Distribution")
            prob_df = pd.DataFrame([res["probabilities"]]).T.reset_index()
            prob_df.columns = ["Sentiment", "Probability"]

            fig_prob = px.bar(
                prob_df,
                x="Sentiment",
                y="Probability",
                color="Sentiment",
                color_discrete_map=COLOR_MAP,
                range_y=[0, 1.0],
                text_auto=".2%",
                title="Model Posterior Probability"
            )
            fig_prob.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True)

            # NLP pipeline inspection
            with st.expander("🛠️ View Internal Preprocessing Pipeline"):
                st.write("**Raw Input:**", user_text)
                st.write("**Cleaned & Lemmatized (Negations preserved):**", res["clean_text"])
                st.write("**Feature Token Prepend:**", f"`{res['aspect']}: {res['clean_text']}`")


# ==============================================================================
# PAGE 4: MODEL PERFORMANCE
# ==============================================================================
elif page == "📈 Model Performance":
    st.markdown('<div class="main-title">📈 Model Evaluation & Benchmark Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Quantitative benchmark of all 5 models evaluated on the student-disjoint test set.</div>', unsafe_allow_html=True)

    comp_csv_path = os.path.join(PROJECT_ROOT, "reports", "model_comparison.csv")
    if os.path.exists(comp_csv_path):
        df_comp = pd.read_csv(comp_csv_path)

        st.subheader("Benchmark Comparison Table")
        # Format table
        st.dataframe(
            df_comp.style.highlight_max(subset=["Macro F1", "Accuracy"], color="#d1fae5"),
            use_container_width=True
        )

        col_m1, col_m2 = st.columns([3, 2])
        with col_m1:
            chart_path = os.path.join(PROJECT_ROOT, "reports", "model_comparison.png")
            if os.path.exists(chart_path):
                st.image(chart_path, caption="Comparative Metric Performance", use_container_width=True)
        with col_m2:
            st.markdown("""
            ### Key Observations:
            - **Best Model**: **TF-IDF + Logistic Regression** achieves the highest Macro F1 (**0.7097**) with strong balanced recall across minority classes.
            - **DistilBERT (Fine-Tuned)** achieved competitive performance (**0.7044 Macro F1**) with the highest Macro Recall (**0.7305**).
            - **Linear SVM** attained the highest raw Accuracy (**79.90%**), but lower recall on the neutral class.
            - **VADER Baseline** scored lowest (**0.6110 Macro F1**), struggling because lexicon lookups lack academic aspect context.
            """)

        st.markdown("---")
        st.subheader("Confusion Matrix Inspector")
        model_choice = st.selectbox(
            "Select Model Confusion Matrix",
            [
                "TF-IDF + Logistic Regression (Best Model)",
                "DistilBERT (Fine-tuned)",
                "TF-IDF + Linear SVM",
                "TF-IDF + Naive Bayes",
                "VADER Baseline"
            ]
        )

        cm_file_map = {
            "TF-IDF + Logistic Regression (Best Model)": "cm_tf_idf_+_logistic_regression.png",
            "DistilBERT (Fine-tuned)": "cm_distilbert_(fine_tuned).png",
            "TF-IDF + Linear SVM": "cm_tf_idf_+_linear_svm.png",
            "TF-IDF + Naive Bayes": "cm_tf_idf_+_naive_bayes.png",
            "VADER Baseline": "cm_vader_baseline.png"
        }

        cm_img = os.path.join(PROJECT_ROOT, "reports", cm_file_map[model_choice])
        if os.path.exists(cm_img):
            st.image(cm_img, width=550)

        st.markdown("---")
        st.subheader("Per-Aspect Performance for Best Model")
        meta_path = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            aspect_f1 = meta.get("per_aspect_f1", {})
            df_asp_f1 = pd.DataFrame(aspect_f1).T.reset_index()
            df_asp_f1.columns = ["Aspect", "Macro F1", "Accuracy", "Test Instances"]
            df_asp_f1["Aspect"] = df_asp_f1["Aspect"].str.replace("_", " ").str.title()
            st.dataframe(df_asp_f1, use_container_width=True)

        st.markdown("---")
        st.subheader("Model Architectures & Principles (Plain English)")
        with st.expander("1. VADER Baseline (Rule-Based Lexicon)", expanded=False):
            st.markdown("""
            - **Mechanism**: Relies on a pre-built dictionary of words mapped to valence scores, with heuristics for capitalizations, punctuation, and modifiers.
            - **Strengths**: Requires zero training data; instant execution.
            - **Weaknesses**: Lacks domain awareness; fails on domain-specific academic vocabulary (e.g. *'attendance'*, *'syllabus'*, *'credits'*).
            """)
        with st.expander("2. TF-IDF + Logistic Regression (Production Best Model)", expanded=True):
            st.markdown("""
            - **Mechanism**: Extracts unigram and bigram word frequencies penalized by document frequency, then fits a logistic sigmoid surface with balanced class weights.
            - **Strengths**: Highly interpretable, fast inference (<2ms), robust generalization without GPU requirement, handles class imbalance gracefully.
            """)
        with st.expander("3. TF-IDF + Multinomial Naive Bayes", expanded=False):
            st.markdown("""
            - **Mechanism**: Applies Bayes theorem assuming feature independence given the class label.
            - **Weaknesses**: Strong independence assumption degrades performance on n-gram collocations and negation phrases.
            """)
        with st.expander("4. TF-IDF + Linear SVM", expanded=False):
            st.markdown("""
            - **Mechanism**: Finds the maximum-margin hyperplane separating classes in high-dimensional TF-IDF space. Calibrated via Platt sigmoid scaling.
            - **Strengths**: High precision on dominant classes; strong decision boundaries.
            """)
        with st.expander("5. Fine-Tuned DistilBERT (Contextual Transformer)", expanded=False):
            st.markdown("""
            - **Mechanism**: 66M parameter distilled bidirectional transformer pre-trained on masked language modeling and fine-tuned for aspect sentiment classification.
            - **Strengths**: Captures complex sentence semantics and deep contextual dependencies.
            """)


# ==============================================================================
# PAGE 5: ABOUT & METHODOLOGY
# ==============================================================================
elif page == "ℹ️ About & Methodology":
    st.markdown('<div class="main-title">ℹ️ About the Project & Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Technical documentation, dataset details, methodology flow, and academic limitations.</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Project Overview
    **Title**: Student Feedback Analysis using NLP (Aspect-Based Sentiment Analysis)  
    **Objective**: Automatically classify student opinions across six critical dimensions of academic life, discover hidden negative themes, and present actionable insights to institutional administrators.

    ---

    ### The 6 Academic Evaluation Aspects
    1. **Teaching**: Pedagogy, teacher demeanor, lecture speed, doubt resolution.
    2. **Course Content**: Curriculum depth, syllabus modernness, theoretical vs practical balance.
    3. **Examination**: Exam difficulty, pattern predictability, time limits, evaluation fairness.
    4. **Lab Work**: Hardware functionality, software licenses, lab assistant guidance.
    5. **Library Facilities**: Textbook availability, latest editions, quiet study zones, Wi-Fi.
    6. **Extracurricular**: Sports equipment, campus clubs, cultural festivals, administrative support.

    ---

    ### End-to-End Methodology Pipeline
    ```
    Raw Student Feedback (student-data.xlsx)
                     │
                     ▼
       Phase 0: Reshape & Validate
       ├─ Long Format (1,324 aspect instances)
       └─ Wide Format (222 students + overall_sentiment)
                     │
                     ▼
       Phase 1 & 2: Preprocessing
       ├─ Clean HTML, URLs, punctuation
       ├─ Preserve negation tokens ("not", "no", "never", "n't")
       └─ WordNet Lemmatization
                     │
                     ▼
       Phase 3: Student-Disjoint Split (70/15/15)
       ├─ Stratified on Student Overall Sentiment
       └─ Zero student leakage across splits
                     │
                     ▼
       Model Training & Comparison
       ├─ VADER Baseline
       ├─ TF-IDF + Logistic Regression (Selected Best)
       ├─ TF-IDF + Naive Bayes
       ├─ TF-IDF + Linear SVM (Calibrated)
       └─ Fine-Tuned DistilBERT (2 Epochs)
                     │
                     ▼
       Phase 4 & 5: Insights & Streamlit Dashboard
    ```

    ---

    ### Limitations & Future Directions
    1. **Metadata Absence**: The dataset does not include course codes, department names, faculty IDs, or dates. Future implementations can introduce temporal trend tracking.
    2. **Short Comment Brevity**: Many students provide single-word replies (*"Good"*, *"Bad"*); context injection via aspect prepending mitigates this.
    3. **Single Institution Sample**: Findings reflect the specific cohort surveyed; cross-institution transfer learning would improve universal generalization.
    """)

