# ============================================================
# TASK 3: SENTIMENT ANALYSIS ON REVIEWS/TWEETS
# Internship AI Task - Kodbud
# Tools: Python, NLTK, scikit-learn
# ============================================================

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────
print("=" * 55)
print("    SENTIMENT ANALYSIS - AI Internship Task 3")
print("=" * 55)

print("\n[Setup] Downloading NLTK resources...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# ─────────────────────────────────────────
# STEP 1: BUILD DATASET
# ─────────────────────────────────────────
print("\n[1/5] Loading dataset...")

# Expanded built-in dataset (positive + negative reviews/tweets)
data = {
    'text': [
        # POSITIVE
        "This product is absolutely amazing! Highly recommend it.",
        "Great quality, fast shipping, very happy with purchase!",
        "Excellent service, the staff were so friendly and helpful.",
        "I love this! Best thing I've bought this year.",
        "Works perfectly, exceeded all my expectations.",
        "Outstanding performance, really impressed by the quality.",
        "Fantastic experience, will definitely buy again.",
        "Super happy with this item, it's exactly what I wanted.",
        "Brilliant product, easy to use and very effective.",
        "This is wonderful, made my life so much easier!",
        "Five stars! Couldn't be more satisfied with this purchase.",
        "Incredible value for money, very impressed overall.",
        "Amazing results, I'm so glad I bought this product.",
        "Lovely design and works like a charm, totally worth it.",
        "Best purchase ever! Exceeded my expectations completely.",
        "Very satisfied, fast delivery and great packaging.",
        "Superb quality, the product looks and feels premium.",
        "Wonderful experience from start to finish, thank you!",
        "Really good product, does exactly what it promises.",
        "So happy I found this, it changed everything for me.",
        # NEGATIVE
        "This is terrible, complete waste of money. Do not buy.",
        "Horrible quality, broke after two days of use.",
        "Very disappointed, nothing like the description at all.",
        "Awful experience, customer service was rude and unhelpful.",
        "Complete rubbish, does not work as advertised at all.",
        "Terrible product, worst purchase I've made in years.",
        "Very poor quality, feels cheap and flimsy in hand.",
        "So frustrated, item arrived damaged and unusable.",
        "Dreadful service, waited 3 weeks and got wrong item.",
        "Absolute garbage, stopped working after first use.",
        "Not happy at all, totally misleading product description.",
        "One star, this product is a scam. Very disappointed.",
        "Shocking quality, falling apart before I even used it.",
        "What a waste! Cheap materials and terrible craftsmanship.",
        "Never buying from this brand again, truly awful product.",
        "Broken on arrival, packaging was damaged and product useless.",
        "Very unhappy, the instructions are confusing and wrong.",
        "Rubbish, doesn't do what it claims, false advertising.",
        "Horrific product, customer service ignored my complaint.",
        "Disappointed with every aspect of this purchase. Never again."
    ],
    'label': ['positive'] * 20 + ['negative'] * 20
}

try:
    url = "https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment/master/Tweets.csv"
    df_raw = pd.read_csv(url)
    df = df_raw[df_raw['airline_sentiment'].isin(['positive','negative'])][['text','airline_sentiment']]
    df.columns = ['text', 'label']
    df = df.dropna().sample(min(2000, len(df)), random_state=42)
    print(f"      ✅ Twitter Airline dataset loaded: {df.shape[0]} records")
except Exception:
    df = pd.DataFrame(data)
    print(f"      ✅ Built-in dataset loaded: {df.shape[0]} records")

print(f"      📊 Positive: {df[df['label']=='positive'].shape[0]}")
print(f"      📊 Negative: {df[df['label']=='negative'].shape[0]}")

# ─────────────────────────────────────────
# STEP 2: TEXT CLEANING
# ─────────────────────────────────────────
print("\n[2/5] Cleaning text data...")

stop_words = set(stopwords.words('english'))
stemmer    = PorterStemmer()

def clean_text(text):
    text = str(text).lower()                          # lowercase
    text = re.sub(r'http\S+|www\S+', '', text)        # remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)             # remove @mentions, #hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)           # keep letters only
    text = re.sub(r'\s+', ' ', text).strip()          # remove extra whitespace
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

df['cleaned'] = df['text'].apply(clean_text)
print("      ✅ Text cleaned: URLs, mentions, stopwords, punctuation removed")

# ─────────────────────────────────────────
# STEP 3: ENCODE LABELS + SPLIT
# ─────────────────────────────────────────
print("\n[3/5] Encoding labels and splitting dataset...")

df['label_num'] = df['label'].map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned'], df['label_num'], test_size=0.2, random_state=42, stratify=df['label_num']
)

# ─────────────────────────────────────────
# STEP 4: TF-IDF + LOGISTIC REGRESSION
# ─────────────────────────────────────────
print("\n[4/5] Vectorizing with TF-IDF & training Logistic Regression...")

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec  = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)
print("      ✅ Model trained successfully!")

# ─────────────────────────────────────────
# STEP 5: EVALUATE
# ─────────────────────────────────────────
print("\n[5/5] Evaluating model...")
y_pred = model.predict(X_test_vec)
acc    = accuracy_score(y_test, y_pred)

print(f"\n      🎯 Accuracy: {acc * 100:.2f}%")
print("\n      📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# ─────────────────────────────────────────
# STEP 6: PREDICT NEW TEXT
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("         🔍 PREDICTING NEW REVIEWS/TWEETS")
print("=" * 55)

new_reviews = [
    "This product is absolutely fantastic! Love it so much.",
    "Worst experience ever. Completely broken on arrival.",
    "It's okay, nothing special but does the job.",
    "I am so happy with this purchase, exceeded expectations!",
    "Terrible customer service, waited weeks and got nothing.",
    "Pretty decent product, good value for the price paid."
]

print()
for review in new_reviews:
    cleaned = clean_text(review)
    vec     = tfidf.transform([cleaned])
    pred    = model.predict(vec)[0]
    proba   = model.predict_proba(vec)[0]
    label   = "😊 POSITIVE" if pred == 1 else "😠 NEGATIVE"
    confidence = max(proba) * 100
    print(f"  {label}  ({confidence:.1f}%)  |  \"{review[:45]}...\"")

print("\n" + "=" * 55)
print("  ✅ Task 3 Complete! Sentiment Analysis is working.")
print("=" * 55)
