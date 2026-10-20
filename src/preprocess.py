"""
src/preprocess.py
Phase 2: Text Preprocessing Module
Exposes clean_text(text) for training, prediction, and Streamlit app.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK resources are available
for resource in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

# Negation words to preserve
NEGATION_WORDS = {"not", "no", "never", "nor", "n't"}

# Build custom stop words set preserving negations
BASE_STOPWORDS = set(stopwords.words("english"))
CUSTOM_STOPWORDS = BASE_STOPWORDS - NEGATION_WORDS

# Initialize WordNet lemmatizer
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Clean and normalize input text:
    1. Handle non-strings and empty input
    2. Lowercase
    3. Remove URLs, emails, and HTML tags
    4. Expand/preserve negation contractions (e.g. don't -> do not)
    5. Remove digits and punctuation
    6. Remove stopwords while keeping negations (not, no, never, nor, n't)
    7. Lemmatize tokens with WordNetLemmatizer
    8. Collapse whitespace and strip
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Convert to lowercase
    cleaned = text.lower()

    # Remove HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    # Remove URLs
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

    # Remove email addresses
    cleaned = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", cleaned)

    # Normalize negation contractions before stripping punctuation
    cleaned = re.sub(r"\bcan['’]t\b", "can not", cleaned)
    cleaned = re.sub(r"\bwon['’]t\b", "will not", cleaned)
    cleaned = re.sub(r"n['’]t\b", " not", cleaned)

    # Remove digits
    cleaned = re.sub(r"\d+", " ", cleaned)

    # Remove punctuation & special characters (preserve letters and whitespace)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)

    # Tokenize by whitespace
    tokens = cleaned.split()

    # Filter stopwords (keeping negations) and lemmatize
    processed_tokens = []
    for token in tokens:
        # Keep negations explicitly
        if token in NEGATION_WORDS:
            processed_tokens.append(token)
        elif token not in CUSTOM_STOPWORDS and len(token) > 1:
            # Lemmatize noun and verb forms
            lemmatized = LEMMATIZER.lemmatize(token)
            processed_tokens.append(lemmatized)

    return " ".join(processed_tokens).strip()


def preprocess_dataset(df, text_col="text", label_col="label", drop_duplicates=True):
    """
    Preprocess a feedback dataframe:
    - Clean text column
    - Drop empty cleaned texts
    - Drop exact duplicates if requested
    """
    df = df.copy()
    df["clean_text"] = df[text_col].apply(clean_text)

    # Drop empty texts
    initial_len = len(df)
    df = df[df["clean_text"].str.strip() != ""].copy()
    dropped_empty = initial_len - len(df)

    if drop_duplicates:
        before_dup = len(df)
        subset_cols = ["clean_text", "aspect"] if "aspect" in df.columns else ["clean_text"]
        if label_col in df.columns:
            subset_cols.append(label_col)
        df = df.drop_duplicates(subset=subset_cols).copy()
        dropped_dups = before_dup - len(df)
    else:
        dropped_dups = 0

    print(f"Preprocessing: {initial_len} -> {len(df)} rows "
          f"(Dropped {dropped_empty} empty, {dropped_dups} duplicates)")
    return df


if __name__ == "__main__":
    sample_texts = [
        "The teaching is NOT good at all!",
        "I don't like the examination pattern, it's very harsh.",
        "Professors are excellent and very helpful :) Check https://college.edu",
        "Lab computers are broken 100% of the time, no software installed.",
        "Neutral feedback, average syllabus."
    ]
    print("Testing clean_text:")
    for st in sample_texts:
        print(f"Original: {st}")
        print(f"Cleaned : {clean_text(st)}\n")
