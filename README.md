Sentiment-Based Product Recommendation & Analysis System
A complete end-to-end system that performs sentiment analysis on product reviews and generates personalized product recommendations based on user behavior and review sentiment. The project includes a Streamlit web app for both sentiment prediction and user-specific recommendations.

🔍 Features
✅ Sentiment Analysis using:

TF-IDF Vectorization (final choice based on best performance)

(GloVe and Word2Vec were tested but not used in final model due to lower accuracy in sentiment prediction tasks)

✅ Machine Learning Models Used:

Random Forest

XGBoost

LightGBM

✅ Final model: Stacking Classifier

Base estimators: RandomForestClassifier, XGBoostClassifier

Meta estimator: LightGBMClassifier

🎯 Achieved 94% accuracy

✅ Robust Preprocessing:

Lowercasing, punctuation & digit removal

Stopword removal

POS-aware lemmatization

✅ Product Recommendation System:

Based on user-product interaction matrix + sentiment scores

Recommends top 5 products for any valid user ID

✅ Streamlit Web App:

🔹 Page 1: User enters a User ID → gets top 5 product recommendations

🔹 Page 2: User enters any review → sentiment is predicted (Positive/Negative)
