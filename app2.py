# -*- coding: utf-8 -*-
"""
Streamlit Sentiment-Based Recommender: TF-IDF + GloVe + VADER + BERT
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
from scipy.sparse import hstack, csr_matrix
from nltk.sentiment import SentimentIntensityAnalyzer

# NLTK downloads
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')

# ---------------- Class Definition ----------------
class SentimentRecommenderModel:
    MODEL_NAME = 'rf_model.joblib'
    VECTORIZER = 'tfidf-vectorizer.pkl'
    RECOMMENDER = 'user_final_rating2.joblib'
    CLEANED_DATA = 'cleaned_data.pkl'
    GLOVE_FILE = 'glove.6B.300d.txt'

    def __init__(self):
        # Load cleaned data
        self.cleaned_data = pickle.load(open(self.CLEANED_DATA, 'rb'))

        # Load ML model
        self.model = joblib.load(open(self.MODEL_NAME, 'rb'))

        # Load TF-IDF vectorizer
        self.vectorizer = joblib.load(open(self.VECTORIZER, 'rb'))
        if not hasattr(self.vectorizer, 'idf_'):
            self.vectorizer.fit(self.cleaned_data["reviews_full_text"].astype(str))

        # Load recommendation data
        self.user_final_rating = joblib.load(open(self.RECOMMENDER, 'rb'))
        self.data = pd.read_csv(r'C:\Users\sayan\OneDrive\Desktop\Amazon, Sentiment analysis + Recommendation system\sample30.csv')

        # Load BERT
        self.bert_tokenizer = AutoTokenizer.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment")
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bert_model.to(self.device)

        # NLP tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.sia = SentimentIntensityAnalyzer()

        # Load GloVe embeddings
        self.embedding_dim = 300
        self.embeddings_index = self.load_glove_embeddings(self.GLOVE_FILE, self.embedding_dim)

    # ------------------- GloVe -------------------
    def load_glove_embeddings(self, file_path, embed_dim=300):
        embeddings = {}
        with open(file_path, encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")
                embeddings[word] = vector
        return embeddings

    def tfidf_weighted_glove(self, text, default_weight=1.0):
        tokens = text.split()
        vecs, weights = [], []
        word2idx = self.vectorizer.vocabulary_

        for w in tokens:
            if w in self.embeddings_index:  # ✅ GloVe available
                if w in word2idx:
                    tfidf_score = self.vectorizer.idf_[word2idx[w]]
                else:
                    tfidf_score = default_weight  # ✅ fallback weight
                vecs.append(self.embeddings_index[w] * tfidf_score)
                weights.append(tfidf_score)

        if len(vecs) > 0:
            doc_vec = np.sum(vecs, axis=0) / (np.sum(weights) + 1e-9)
        else:
            doc_vec = np.zeros(self.embedding_dim)

        return doc_vec

    # ------------------- Build Feature Vector -------------------
    def build_feature_vector(self, texts):
        """Builds TF-IDF + GloVe + VADER feature vector for a list of texts"""
        clean_texts = [self.preprocess_text(t) for t in texts]

        # TF-IDF
        tfidf_vec = self.vectorizer.transform(clean_texts)

        # GloVe
        glove_vecs = np.array([self.tfidf_weighted_glove(t) for t in clean_texts])
        glove_sparse = csr_matrix(glove_vecs)

        # VADER
        vader_vecs = []
        for t in texts:
            s = self.sia.polarity_scores(t)
            vader_vecs.append([s["neg"], s["pos"], s["compound"]])
        vader_sparse = csr_matrix(np.array(vader_vecs))

        return hstack([tfidf_vec, glove_sparse, vader_sparse])

    # ------------------- Recommendation -------------------
    def getRecommendationByUser(self, user):
        return list(self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)

    # ------------------- Hybrid ML Prediction -------------------
    def classify_sentiment_hybrid(self, review_text, vader_threshold=0.3):
        combined_vec = self.build_feature_vector([review_text])
        ml_pred = self.model.predict(combined_vec)[0]

        vader_scores = self.sia.polarity_scores(review_text)
        compound = vader_scores["compound"]

        if compound >= vader_threshold:
            final_pred = 1
        elif compound <= -vader_threshold:
            final_pred = 0
        else:
            final_pred = ml_pred

        return final_pred

    # ------------------- BERT Prediction -------------------
    def classify_sentiment_bert(self, review_text):
        inputs = self.bert_tokenizer(review_text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        score = torch.argmax(probs).item() + 1
        labels = {1: "Very Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Very Positive"}
        return labels[score]

    # ------------------- Preprocessing -------------------
    def preprocess_text(self, text):
        text = text.lower().strip()
        text = re.sub("\[\s*\w*\s*\]", "", text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub("\S*\d\S*", "", text)
        return self.lemma_text(text)

    def remove_stopword(self, text):
        return " ".join([w for w in text.split() if w.isalpha() and w not in self.stop_words])

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
# ---------------- Streamlit App ----------------
st.set_page_config(page_title="Sentiment-Based Recommender", layout="centered")
st.markdown("""
    <style>
    
    /* ---------- Full Page Lighter Gradient Background ---------- */
    body, .stApp {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        background-attachment: fixed;
        color: #333;
        font-family: 'Helvetica', sans-serif;
    }

    /* ---------- Animated Buttons ---------- */
    .stButton>button {
        background: linear-gradient(90deg, #89f7fe, #66a6ff);
        color: #fff;
        font-size: 16px;
        font-weight: bold;
        border-radius: 12px;
        padding: 0.6em 1.2em;
        border: none;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #66a6ff, #89f7fe);
        box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
    }

    /* ---------- TextArea Styling ---------- */
    textarea {
        border-radius: 12px;
        padding: 0.5em;
        font-size: 14px;
        border: 2px solid #66a6ff;
        transition: all 0.3s ease;
    }
    textarea:focus {
        border-color: #89f7fe;
        box-shadow: 0 0 10px rgba(137, 247, 254, 0.5);
    }

    /* ---------- Table Styling ---------- */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        color: #000;
        background-color: #ffffffcc;
    }

    /* ---------- Tabs Styling ---------- */
    div[role="tablist"] button {
        background: linear-gradient(90deg, #89f7fe, #66a6ff);
        color: #fff;
        font-weight: bold;
        border-radius: 12px 12px 0 0;
        transition: all 0.3s ease;
    }
    div[role="tablist"] button:hover {
        background: linear-gradient(90deg, #66a6ff, #89f7fe);
        transform: scale(1.05);
    }

    /* ---------- Titles ---------- */
    h1, h2, h3, h4, h5 {
        color: #333;
    }

    /* ---------- Footer ---------- */
    footer {visibility: hidden;}
    .custom-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        font-size: 14px;
        color: #333;
        padding: 10px;
        background: rgba(255, 255, 255, 0.6);
    }

    </style>

    <div class="custom-footer">
        Made by Sayan Banerjee
    </div>
""", unsafe_allow_html=True)


st.title("🛒 Sentiment-Based Product Recommender")
st.markdown("Analyze product reviews with advanced sentiment detection and get personalized product recommendations based on your preferences.")



# ---------------- Cached model loader (only change) ----------------
@st.cache_resource
def load_sentiment_model():
    return SentimentRecommenderModel()

# Use cached model so heavy objects (GloVe, models, BERT) load only once
model = load_sentiment_model()

tab1, tab2 = st.tabs(["📌 Recommend by User", "📝 Classify Review"])

# --- Tab 1: Recommend Products with Dropdown ---
with tab1:
    user_ids = model.user_final_rating.index.tolist()
    user_input = st.selectbox("Select your User ID:", options=user_ids)

    if st.button("Get Top 5 Recommendations"):
        if user_input:
            recommendations = model.getRecommendationByUser(user_input)
            filtered_data = model.cleaned_data[model.cleaned_data.id.isin(recommendations)]

            # ✅ FIX: use hybrid feature vector instead of only TF-IDF
            X = model.build_feature_vector(filtered_data["reviews_full_text"].astype(str).tolist())
            filtered_data["predicted_sentiment"] = model.model.predict(X)

            temp = filtered_data[['id', 'predicted_sentiment']]
            temp_grouped = temp.groupby('id', as_index=False).count()
            temp_grouped["pos_review_count"] = temp_grouped.id.apply(
                lambda x: temp[(temp.id == x) & (temp.predicted_sentiment == 1)]["predicted_sentiment"].count())
            temp_grouped["total_review_count"] = temp_grouped['predicted_sentiment']
            temp_grouped['pos_sentiment_percent'] = np.round(
                temp_grouped["pos_review_count"] / temp_grouped["total_review_count"] * 100, 2)
            sorted_products = temp_grouped.sort_values('pos_sentiment_percent', ascending=False)[0:5]
            result = pd.merge(model.data, sorted_products, on="id")[["name", "brand", "manufacturer", "pos_sentiment_percent"]].drop_duplicates().sort_values(['pos_sentiment_percent', 'name'], ascending=[False, True])
            st.dataframe(result)
        else:
            st.warning("Please select a valid user ID.")

# --- Tab 2: Classify Review ---
with tab2:
    review_text = st.text_area("Enter a product review:")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("TF-IDF + GloVe + VADER"):
            if review_text.strip():
                try:
                    prediction = model.classify_sentiment_hybrid(review_text)
                    if prediction == 1:
                        st.success("✅ Positive Review (ML + GloVe + VADER)")
                    else:
                        st.error("❌ Negative Review (ML + GloVe + VADER)")
                except Exception as e:
                    st.error(f"Hybrid model error: {e}")
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


