# 🛒 Sentiment-Based Product Recommendation & Analysis System

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)  
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)  
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-green?logo=scikitlearn)  
![TensorFlow](https://img.shields.io/badge/DL-TensorFlow-orange?logo=tensorflow)  
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-purple)  
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📖 Project Overview
The **Sentiment-Based Product Recommendation & Analysis System** is an **end-to-end AI solution** that combines **sentiment analysis** of user reviews with **personalized product recommendations**.  

- Built with **Machine Learning, NLP, and Recommendation Systems**  
- Deployed using **Streamlit** for interactive user experience  
- Designed to help **businesses and e-commerce platforms** better understand customer opinions and recommend products accordingly.  

---

## 🔍 Features

### ✅ Sentiment Analysis
- **Text Preprocessing**:  
  - Lowercasing, punctuation & digit removal  
  - Stopword removal  
  - POS-aware lemmatization  

- **Vectorization Approaches**:  
  - TF-IDF (final choice based on performance)  
  - GloVe and Word2Vec (tested but not used in final deployment)  

- **Models Evaluated**:  
  - Random Forest  
  - XGBoost  
  - LightGBM  

- **Final Model**:  
  - **Stacking Classifier** with:  
    - Base Estimators → RandomForest, XGBoost  
    - Meta Estimator → LightGBM  
  - **Achieved 94% Accuracy** 🎯  

---

### ✅ Product Recommendation System
- **Hybrid Recommendation Engine**:  
  - User-Product interaction matrix  
  - Integrated with sentiment polarity scores  
- **Generates Top-5 Personalized Product Recommendations** for any valid user ID  

---

### ✅ Streamlit Web Application
1. **Recommendation Page**  
   - User enters a valid **User ID**  
   - System outputs **Top 5 recommended products**  
2. **Sentiment Prediction Page**  
   - User enters a **product review**  
   - Model predicts **Positive / Negative sentiment**  

📌 **Live Deployment Links:**  
- 🌐 [Streamlit Web App](https://sentiment-recommendation-sayan.streamlit.app/)  
- 🎥 [Live Demo (Google Drive)](https://drive.google.com/file/d/1Wfur4J0WadaKk6398mQ9yxaDlWxWYEWX/view?usp=sharing)  

---

## 🧠 Advanced NLP Extension (BERT Pipeline)
In addition to the ML-based models, a **BERT-based sentiment classification pipeline** was developed and deployed on Streamlit.  

- Used **HuggingFace Transformers** for BERT fine-tuning  
- Achieved higher generalization across **unseen and complex reviews**  
- Provides **deep contextual understanding** beyond TF-IDF/Word2Vec  
- Streamlit app allows switching between **Traditional ML model** and **BERT-based model**  

---

## 📊 Tech Stack

| Category              | Tools / Libraries |
|-----------------------|------------------|
| **Frontend**          | Streamlit |
| **ML Models**         | Scikit-learn (RF, XGBoost, LightGBM) |
| **Deep Learning (NLP)** | HuggingFace Transformers (BERT), TensorFlow |
| **Vectorization**     | TF-IDF, GloVe, Word2Vec |
| **Visualization**     | Matplotlib, Seaborn |
| **Deployment**        | Streamlit Cloud |

---

## 📁 Project Structure
├── app.py # Streamlit app
├── models/ # Trained ML/DL models
│ ├── stacking_model.pkl
│ ├── bert_model/
├── utils/ # Helper functions
│ ├── preprocess.py
│ ├── recommend.py
│ ├── sentiment.py
├── data/ # Sample datasets
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## 🚀 Future Improvements
- [ ] Enhance product recommendations using **Hybrid Collaborative Filtering + BERT embeddings**  
- [ ] Add **Explainable AI (LIME/SHAP)** for better interpretability of sentiment predictions  
- [ ] Multi-language support for non-English product reviews  
- [ ] Integration with **real-time e-commerce APIs**  
- [ ] Deploy mobile-friendly version with **FastAPI + Flutter/React Native**  

---

## 🤝 Contributing
Contributions are welcome! 🎉  

1. Fork the repository  
2. Create a new feature branch (`feature-new`)  
3. Commit your changes  
4. Push the branch  
5. Open a Pull Request  

---

## 💖 Show Your Support
If you found this project helpful:  
⭐ Star the repository  
🔗 Share with your peers  
📢 Mention it on LinkedIn / GitHub  

---

## 👨‍💻 Developed By
**Sayan Banerjee**  
🎓 MSc in Statistics & Computing, BHU  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sayan%20Banerjee-blue?logo=linkedin)](https://github.com/Sayan-ML) 
[![GitHub](https://img.shields.io/badge/GitHub-SayanBanerjee-black?logo=github)](https://www.linkedin.com/in/sayan-banerjee-0222a4214/)  

---

## 📄 License
This project is licensed under the **MIT License** – you are free to use, modify, and distribute it with attribution.  

