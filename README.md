# AG News Category Classification

This project is part of my **one-month NLP internship at Elevvo**.  
It demonstrates **multiclass text classification, text preprocessing, feature engineering, and machine learning/deep learning on news articles**.

---

## 📝 Project Overview

- **Dataset:** AG News (Kaggle)  
- **Goal:** Classify news articles into categories: **Business, Sports, Politics, Technology**  
- **Techniques Used:**
  - Text preprocessing: lowercasing, punctuation removal, tokenization, stopword removal, lemmatization
  - Feature extraction: TF-IDF vectorization (optimized for speed)
  - Machine learning models: Linear SVM, Logistic Regression, LightGBM, XGBoost (CPU optimized)
  - Deep learning: Feedforward Neural Network using TensorFlow/Keras (GPU enabled)
  - Model evaluation: Accuracy, classification report, confusion matrix
  - Visualization: Top frequent words per category

---

## ⚙️ Technologies & Libraries

- Python, Pandas, NumPy, Matplotlib, Seaborn  
- Scikit-learn: LinearSVC, Logistic Regression  
- XGBoost, LightGBM  
- NLTK: Stopwords, Lemmatization  
- TensorFlow/Keras: Neural Network  
- WordCloud for visualization  

---

## 📈 Model Performance

| Model | Accuracy |
|-------|----------|
| **Linear SVM** | 91.31% |
| **Logistic Regression** | 91.48% |
| **LightGBM** | 90.25% |
| **XGBoost (CPU)** | 86.47% |
| **Neural Network (GPU)** | 91.13% |

**Observations:**

- Linear SVM and Logistic Regression are the top-performing models (~91–92%).  
- LightGBM performs slightly lower (~90%).  
- XGBoost in CPU mode is slower and achieves lower accuracy (~86%) due to large dataset and no GPU acceleration in Kaggle.  
- Neural Network achieves comparable results (~91%) and leverages GPU for faster training.

---

## 🖼 Visualizations

### Confusion Matrix Example
![Confusion Matrix]("C:\Users\EGY10\Pictures\Screenshots\Screenshot 2026-02-23 193617.png")

### Top Frequent Words per Category
![Top Words]("C:\Users\EGY10\Pictures\Screenshots\Screenshot 2026-02-23 193617.png")

> *Tip:* Replace the placeholders with your actual screenshots from Kaggle.

---

## 🔗 Links

- Kaggle Notebook: [AG News Classification](https://www.kaggle.com/code/ahmedbakr888/notebook2e6ccb6626)  
- GitHub Repository: [Ahmedbakr888 NLP Projects](https://github.com/Ahmedbakr888/imdb-sentiment-analysis)

---

## 📌 Next Steps

- Hyperparameter tuning for SVM, Logistic Regression, and Neural Network  
- Explore **Transformer models** (BERT, DistilBERT) for higher accuracy  
- Deploy as a web app using **Streamlit** or **Flask**  
- Add advanced visualizations for category insights and word importance  

---

**Author:** Ahmed Bakr  
**Elevvo NLP Internship | One-Month Program**
