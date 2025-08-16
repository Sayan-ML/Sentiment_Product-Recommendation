# -*- coding: utf-8 -*-
"""
Streamlit Sentiment-Based Recommender with TF-IDF, ML, XGBoost, LightGBM, and BERT
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import nltk
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import xgboost as xgb
import lightgbm as lgb

# NLTK downloads
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ---------------- Class Definition ----------------
class SentimentRecommenderModel:
    MODEL_NAME = 'stacking_model_compressed.joblib'
    VECTORIZER = 'tfidf-vectorizer.pkl'
    RECOMMENDER = 'user_final_rating2.joblib'
    CLEANED_DATA = 'cleaned_data.pkl'

    def __init__(self):
        # --- Load ML model ---
        self.model = joblib.load(open(self.MODEL_NAME, 'rb'))

        # --- Load TF-IDF Vectorizer ---
        self.vectorizer = joblib.load(open(self.VECTORIZER, 'rb'))
        if not hasattr(self.vectorizer, 'idf_'):
            # Fit on cleaned data if IDF missing
            self.cleaned_data = pickle.load(open(self.CLEANED_DATA, 'rb'))
            self.vectorizer.fit(self.cleaned_data["reviews_full_text"].astype(str))

        # --- Load recommendation data ---
        self.user_final_rating = joblib.load(open(self.RECOMMENDER, 'rb'))
        self.data = pd.read_csv(r'C:\Users\sayan\OneDrive\Desktop\Amazon, Sentiment analysis + Recommendation system\sample30.csv')

        # --- Load BERT model ---
        self.bert_tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.bert_model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bert_model.to(self.device)

        # --- Load ML boosting models if needed ---
        # Example: self.xgb_model = joblib.load("xgb_model.pkl")
        # Example: self.lgb_model = joblib.load("lgb_model.pkl")

        # --- NLP preprocessing ---
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    # ---------------- Recommendation ----------------
    def getRecommendationByUser(self, user):
        return list(self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)

    def getSentimentRecommendations(self, user):
        if user in self.user_final_rating.index:
            recommendations = self.getRecommendationByUser(user)
            filtered_data = self.cleaned_data[self.cleaned_data.id.isin(recommendations)]
            X = self.vectorizer.transform(filtered_data["reviews_full_text"].astype(str))
            filtered_data["predicted_sentiment"] = self.model.predict(X)
            temp = filtered_data[['id', 'predicted_sentiment']]
            temp_grouped = temp.groupby('id', as_index=False).count()
            temp_grouped["pos_review_count"] = temp_grouped.id.apply(
                lambda x: temp[(temp.id == x) & (temp.predicted_sentiment == 1)]["predicted_sentiment"].count())
            temp_grouped["total_review_count"] = temp_grouped['predicted_sentiment']
            temp_grouped['pos_sentiment_percent'] = np.round(
                temp_grouped["pos_review_count"] / temp_grouped["total_review_count"] * 100, 2)
            sorted_products = temp_grouped.sort_values('pos_sentiment_percent', ascending=False)[0:5]
            return pd.merge(self.data, sorted_products, on="id")[["name", "brand", "manufacturer", "pos_sentiment_percent"]].drop_duplicates().sort_values(['pos_sentiment_percent', 'name'], ascending=[False, True])
        else:
            return None

    # ---------------- ML Sentiment ----------------
    def classify_sentiment_ml(self, review_text):
        review_text = self.preprocess_text(review_text)
        X = self.vectorizer.transform([review_text])
        return self.model.predict(X)[0]

    # ---------------- BERT Sentiment ----------------
    def classify_sentiment_bert(self, review_text):
        inputs = self.bert_tokenizer(review_text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        score = torch.argmax(probs).item() + 1
        labels = {1: "Very Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Very Positive"}
        return labels[score]

    # ---------------- Preprocessing ----------------
    def preprocess_text(self, text):
        text = text.lower().strip()
        text = re.sub("\[\s*\w*\s*\]", "", text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub("\S*\d\S*", "", text)
        return self.lemma_text(text)

    def remove_stopword(self, text):
        return " ".join([word for word in text.split() if word.isalpha() and word not in self.stop_words])

    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def lemma_text(self, text):
        word_pos_tags = nltk.pos_tag(word_tokenize(self.remove_stopword(text)))
        words = [self.lemmatizer.lemmatize(tag[0], self.get_wordnet_pos(tag[1])) for tag in word_pos_tags]
        return " ".join(words)

# ---------------- Streamlit App ----------------
st.set_page_config(page_title="Sentiment-Based Recommender", layout="centered")
st.title("🛒 Sentiment-Based Product Recommender")
st.markdown("Get product recommendations based on sentiment analysis of reviews.")

model = SentimentRecommenderModel()

tab1, tab2 = st.tabs(["📌 Recommend by User", "📝 Classify Review"])

# --- Tab 1: Recommend Products ---
with tab1:
    user_input = st.text_input("Enter your User ID:")
    if st.button("Get Top 5 Recommendations"):
        if user_input:
            result = model.getSentimentRecommendations(user_input)
            if result is not None and not result.empty:
                st.success("Top 5 Recommended Products Based on Positive Sentiment:")
                st.dataframe(result.reset_index(drop=True))
            else:
                st.warning("User not found or no recommendations available.")
        else:
            st.warning("Please enter a valid user ID.")

# --- Tab 2: Classify Review ---
with tab2:
    review_text = st.text_area("Enter a product review:")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("TF-IDF + ML"):
            if review_text.strip():
                try:
                    prediction = model.classify_sentiment_ml(review_text)
                    if prediction == 1:
                        st.success("✅ Positive Review (ML)")
                    else:
                        st.error("❌ Negative Review (ML)")
                except Exception as e:
                    st.error(f"ML model error: {e}")
            else:
                st.warning("Please enter a valid review.")

    with col2:
        if st.button("BERT"):
            if review_text.strip():
                try:
                    prediction = model.classify_sentiment_bert(review_text)
                    st.info(f"📝 Sentiment by BERT: {prediction}")
                except Exception as e:
                    st.error(f"BERT model error: {e}")
            else:
                st.warning("Please enter a valid review.")
