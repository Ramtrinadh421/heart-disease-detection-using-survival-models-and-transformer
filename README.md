# ❤️ Heart Disease Risk Prediction Using Survival Models & Transformers  
### Team Capstone Project – VIT-AP University

---

## 📌 Overview
This repository contains our team capstone project focused on predicting the **10-year risk of Coronary Heart Disease (CHD)** using a combination of **Survival Analysis** and **Transformer-based deep learning techniques**. Using the Framingham Heart Study dataset, we built a complete end-to-end system—from preprocessing and modeling to deployment through a Streamlit web application.

Our goal was to design a clinically interpretable model capable of estimating long-term risk and presenting real-time predictions.

---

## 👥 Team Capstone Project Description
This project was collaboratively developed as part of our final-year capstone. Our team worked across:

- Data cleaning & preprocessing  
- Exploratory Data Analysis  
- Survival label creation (event + time-to-event)  
- FT-Transformer feature representation  
- Cox Proportional Hazards survival modeling  
- Model evaluation (C-index, hazard ratios, KM curves)  
- SHAP interpretability  
- Streamlit web application development  

The final outcome is a **scalable ML system** for real-time heart disease risk prediction.

---

## 🚀 Key Features
- End-to-end machine learning pipeline  
- FT-Transformer for deep tabular learning  
- Cox Proportional Hazards model  
- 10-year CHD risk scoring  
- C-Index evaluation  
- SHAP + KM Curve interpretability  
- Streamlit app for real-time predictions  
- Modular Python codebase

---

## 🧠 Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Lifelines (Cox Model)  
- PyTorch (FT-Transformer)  
- Matplotlib, Seaborn  
- SHAP  
- Streamlit  

---

## 📁 Project Structure
```
heart-disease-survival-analysis/
│── README.md
│── requirements.txt
│
├── data/
│
├── notebooks/
│   ├── eda.ipynb
│   └── modeling.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── survival_model.py
│   ├── transformer_model.py
│   └── utils.py
│
├── app/
│   ├── app.py
│   └── assets/
│
└── models/
    ├── cox_model.pkl
    └── transformer_model.pt
```

---

## 🔄 Workflow / Methodology

### 1️⃣ Data Preprocessing  
- Missing value handling  
- Encoding + scaling  
- Outlier handling  
- Creating survival labels:  
  - **event:** CHD occurrence  
  - **time:** follow-up duration  

### 2️⃣ Feature Engineering  
- FT-Transformer embeddings  
- Deep feature tokenization  

### 3️⃣ Modeling  
- Cox Proportional Hazards survival model  
- Risk score computation  

### 4️⃣ Evaluation  
- C-Index  
- KM survival curves  
- Hazard Ratios  
- SHAP explainability  

### 5️⃣ Deployment  
- Streamlit Web App  
- Single + batch prediction support  

---

## 📊 Results
- **C-Index:** ~0.75–0.80  
- KM curves show clear risk-group separation  
- Top predictors (SHAP):  
  - Age  
  - Cholesterol  
  - BP  
  - Smoking  
  - Diabetes indicators  

---

## ▶️ Running the Streamlit App
```
cd app
streamlit run app.py
```

---

## 🔧 Installation
```
git clone https://github.com/your-username/heart-disease-survival-analysis.git
cd heart-disease-survival-analysis
pip install -r requirements.txt
```

---

## 📥 Dataset
Framingham Heart Study Dataset  
Download link: [Kaggle – Framingham Heart Study Dataset](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset)

---


## ⭐ If you like this project, please give it a star!
