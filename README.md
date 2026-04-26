# E-Commerce Customer Churn Prediction

Predicting whether an e-commerce customer will ever make a second purchase, using real commercial transaction data — 100,000 orders across 9 relational tables, structured similarly to how Shopee, Lazada, and Tiki operate internally.

🔗 **[Live Demo](https://ecommerce-churn-prediction-6rm7slrglgsuwbvvvahear.streamlit.app/)**

---

## The Problem

97% of customers never came back after their first purchase. That's not a data quirk — it's a structural retention problem that costs e-commerce businesses significant revenue every month. This project builds a machine learning system to identify which customers are likely to churn, explain why it's happening, and give the business something actionable to work with.

---

## What I Found

Delivery delay is the single biggest driver of churn — not price, not product quality, not review scores. Customers who received their orders late almost never returned. Customers who received orders early were significantly more likely to buy again. That one finding reframes where an e-commerce company should invest first: operations and logistics, before marketing and discounts.

A secondary finding was that customers who bought large, heavy items — furniture, appliances — almost never returned regardless of their delivery experience. These are one-off purchases by nature, and treating them the same as regular buyers in a retention campaign wastes budget.

---

## How I Built It

I started by merging six relational tables to build a single customer-level feature table — the same kind of data wrangling a Data Scientist at Shopee or Lazada would do daily. From there I engineered 21 features across four categories: purchase recency and spend, delivery experience, payment behavior, and review signals.

I trained three models in order of complexity — logistic regression as a baseline, then LightGBM, then LightGBM tuned with Optuna's Bayesian hyperparameter search across 50 trials. The final model reached an AUC-ROC of 0.9991. Class imbalance was handled with class weights rather than oversampling, to avoid introducing artificial patterns into the training data.

For explainability I used SHAP to break down what drove each prediction — both globally across all customers and for individual cases. This is what makes the model useful to a business rather than just accurate on paper.

---

## The Dashboard

The Streamlit app has three sections. The overview page summarizes the problem, the data, and the key findings for anyone who wants the business context first. The churn predictor lets you input a customer's details and returns a real-time churn probability, a risk level, a plain English business recommendation, and a SHAP breakdown of the top factors driving that specific prediction. The model insights page shows the global SHAP charts, ROC curve, and model comparison for a more technical audience.

---

## Results

The tuned LightGBM model achieved an AUC-ROC of 0.9991, up from 0.9956 for the logistic regression baseline. More importantly, it correctly identifies 98% of retained customers — meaning businesses can trust the model to flag who is actually worth targeting in a retention campaign without wasting budget on customers who were never going to return.

---

## Tech Stack

Python, pandas, numpy, LightGBM, Optuna, SHAP, Streamlit, Docker, GitHub Codespaces.

---

## Run It Locally

```bash
git clone https://github.com/yourusername/ecommerce-churn-prediction
cd ecommerce-churn-prediction
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place the CSV files in the `data/` folder.

---

Built as a portfolio project targeting Data Scientist roles in Singapore and Vietnam.
