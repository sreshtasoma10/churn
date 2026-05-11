# Customer Churn Analysis & Prediction System

> End-to-end machine learning pipeline for telecom customer churn prediction — with a live interactive dashboard, dual input modes, and pincode-level retention analytics.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=flat&logo=streamlit)](https://jknprek7zvy7wrds7axc25.streamlit.app/)
[![Research Paper](https://img.shields.io/badge/Published-SSRN-1D6F9C?style=flat)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5222370)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What this project does

Telecom companies lose significant revenue to customer churn. This system predicts which customers are likely to churn — and explains *why* — using a combination of machine learning models and an interpretable rule-based classifier.

**Key capabilities:**
- Predicts churn probability per customer using K-NN and Random Forest models
- Applies a custom rule-based classifier for transparent, human-readable decision logic
- Accepts input via CSV file upload **or** direct MySQL database connection
- Visualises churn patterns at pincode level using Power BI-style analytics
- Deployed as a live Streamlit dashboard — no local setup needed

---

## Live Demo

**[Open the app →](https://jknprek7zvy7wrds7axc25.streamlit.app/)**

Upload a customer CSV or connect to a database and get real-time churn predictions with visual insights.

---

## Research Publication

This project is accompanied by a peer-reviewed research paper published on SSRN:

**[Customer Churn Analysis & Telecom Prediction System — Read the paper →](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5222370)**

---

## Tech Stack

| Layer | Tools |
|---|---|
| Machine Learning | Scikit-learn, Pandas, NumPy |
| Database Integration | SQLAlchemy, MySQL |
| Dashboard & UI | Streamlit, Matplotlib |
| Business Analytics | Power BI |
| Language | Python 3.10+ |

---

## Project Architecture

```
churn/
├── deep.py               # Core ML pipeline — model training, evaluation, rule classifier
├── requirements.txt      # Dependencies
└── README.md
```

**Pipeline flow:**

```
Raw Data (CSV / MySQL)
        ↓
Preprocessing & Feature Engineering
        ↓
Model Training (K-NN + Random Forest + Rule Classifier)
        ↓
Churn Prediction + Probability Scores
        ↓
Streamlit Dashboard (real-time predictions + pincode analytics)
```

---

## Models Used

### 1. K-Nearest Neighbours (K-NN)
Baseline classifier. Identifies customers similar to known churners based on usage patterns and demographics.

### 2. Random Forest
Ensemble model. Handles non-linear relationships and feature interactions. Provides feature importance scores for interpretability.

### 3. Custom Rule-Based Classifier
Built from scratch to make predictions transparent. Converts model logic into human-readable rules — e.g., *"If contract type = month-to-month AND tenure < 6 months AND support tickets > 3 → high churn risk."* Designed for business teams who need explainable decisions, not just accuracy scores.

---

## Key Features

**Dual input modes**
Connect directly to a MySQL database via SQLAlchemy, or upload a structured CSV — both produce identical outputs.

**Pincode-level churn analytics**
Breaks down churn rates by geography, enabling targeted regional retention campaigns.

**Real-time predictions**
The Streamlit dashboard processes new customer data on the fly and returns churn probability scores with visual breakdowns.

---

## Setup (run locally)

```bash
git clone https://github.com/sreshtasoma10/churn.git
cd churn
pip install -r requirements.txt
streamlit run deep.py
```

For MySQL input mode, configure your DB credentials in the app sidebar when it launches.

---

## Results

- Random Forest outperformed K-NN on precision for minority class (churners)
- Rule-based classifier achieved comparable accuracy while providing full decision transparency
- Pincode-level segmentation revealed geographic churn clusters actionable for retention teams

---

## Author

**Sreshta Soma**
[LinkedIn](https://linkedin.com/in/sreshtasoma10) · [Portfolio](https://datascienceportfol.io/somasreshta) · [GitHub](https://github.com/sreshtasoma10)
