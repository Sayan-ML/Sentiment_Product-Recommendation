import streamlit as st
import os
import pickle
import joblib
import pandas as pd
import numpy as np
import nltk
import string
import re
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer




# ========= Streamlit App =========
st.set_page_config(page_title="Sentiment-Based Recommender", layout="centered")

# --- Custom CSS for Dark Theme & Stylish UI ---
st.markdown("""
    <style>
        /* Main background and text */
        body, .stApp {
            background-color: #0E1117;
            color: white;
        }
        /* Titles */
        h1, h2, h3, h4 {
            color: #FAFAFA;
        }
        /* Buttons */
        div.stButton > button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
            font-size: 16px;
            font-weight: bold;
            border: none;
            transition: 0.3s;
        }
        div.stButton > button:hover {
            background-color: #45a049;
            transform: scale(1.02);
        }
        /* Selectbox */
        .stSelectbox div[data-baseweb="select"] {
            background-color: #262730;
            color: white;
            border-radius: 8px;
        }
        /* Text area */
        textarea {
            background-color: #262730 !important;
            color: white !important;
            border-radius: 8px !important;
        }
        /* Dataframe table dark mode */
        .stDataFrame {
            background-color: #1E1E1E;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header Banner ---
st.markdown("""
<div style='background-color:#262730; padding:15px; border-radius:10px; text-align:center'>
    <h1 style='color:white;'>🛒 Sentiment-Based Product Recommender</h1>
    <p style='color:lightgray; font-size:16px;'>
        Get product recommendations based on <b>sentiment analysis</b> of reviews.
    </p>
</div>
""", unsafe_allow_html=True)

# Load model
try:
    model = SentimentRecommenderModel()
except Exception:
    st.stop()

# --- Tabs ---
tab1, tab2 = st.tabs(["📌 Recommend by User", "📝 Classify Review"])

# --- Tab 1: Recommend Products ---
with tab1:
    st.markdown("### 🔍 Select Your User ID")
    user_ids = sorted(model.user_final_rating.index.tolist())
    user_input = st.selectbox("", user_ids)

    if st.button("Get Top 5 Recommendations"):
        result = model.getSentimentRecommendations(user_input)
        if result is not None and not result.empty:
            st.success("Top 5 Recommended Products Based on Positive Sentiment:")
            st.dataframe(result.reset_index(drop=True))
        else:
            st.warning("No recommendations available for this user.")

# --- Tab 2: Classify Review ---
with tab2:
    st.markdown("### ✍️ Enter a product review for sentiment classification")
    review_text = st.text_area("")
    if st.button("Predict Sentiment"):
        if review_text:
            prediction = model.classify_sentiment(review_text)
            if prediction[0] == 1:
                st.success("✅ Positive Review")
            else:
                st.error("❌ Negative Review")
        else:
            st.warning("Please enter a valid review.")



