import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import lightgbm as lgb

# page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🛒",
    layout="wide"
)

# load model
@st.cache_resource
def load_model():
    with open('../models/lgbm_tuned.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()
explainer = shap.TreeExplainer(model)

# feature list — same order as training
FEATURES = [
    'recency', 'monetary', 'avg_delivery_delay', 'max_delivery_delay',
    'late_deliveries', 'avg_installments', 'unique_payment_types',
    'used_voucher', 'avg_item_price', 'max_item_price',
    'avg_product_weight', 'avg_freight_value', 'avg_review_score',
    'min_review_score', 'gave_bad_review'
]

# sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Churn predictor", "Model insights"])

# ── page 1: overview ──────────────────────────────────────────
if page == "overview":
    st.title("🛒 E-Commerce Customer Churn Prediction")
    st.markdown("""
    This project predicts whether an e-commerce customer will ever make a second purchase,
    using real transaction data from Olist — a Brazilian marketplace with a customer behavior
    pattern very similar to Shopee, Lazada, and Tiki in Southeast Asia.
    """)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("total customers", "55,364")
    col2.metric("churn rate", "96%")
    col3.metric("model auc-roc", "0.9991")

    st.markdown("---")
    st.subheader("key finding")
    st.info("""
    **Delivery delay — not price, not product quality — is the #1 driver of customer churn.**
    Customers who received orders late almost never returned. Customers who received orders
    early were significantly more likely to buy again.
    """)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("churn distribution")
        st.markdown("""
        96% of customers never made a second purchase. Only 4% returned for another order.
        This extreme imbalance is the core business problem — almost every customer is a
        one-time buyer, making retention campaigns critical for revenue growth.
        """)
        st.image('../data/churn_distribution.png', width=350)

    with col2:
        st.subheader("orders over time")
        st.markdown("""
        Order volume grew steadily from late 2016 through mid 2018, with a sharp spike
        around January 2018 — likely driven by a seasonal sales event. Despite growing
        order numbers, the churn rate remained consistently high throughout the period,
        confirming this is a structural retention problem, not a temporary one.
        """)
        st.image('../data/orders_over_time.png', width=450)

# ── page 2: churn predictor ───────────────────────────────────
elif page == "churn predictor":
    st.title("🔍 Customer Churn Predictor")
    st.markdown("Enter a customer's details to predict whether they will return for a second purchase.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("purchase behavior")
        recency = st.slider("days since last purchase", 0, 730, 200)
        monetary = st.number_input("total spend (BRL)", 0.0, 5000.0, 150.0, step=10.0)
        avg_item_price = st.number_input("average item price (BRL)", 0.0, 2000.0, 100.0, step=10.0)
        max_item_price = st.number_input("most expensive item (BRL)", 0.0, 5000.0, 150.0, step=10.0)

    with col2:
        st.subheader("delivery experience")
        avg_delivery_delay = st.slider("average delivery delay (days)", -30, 60, 0)
        max_delivery_delay = st.slider("worst delivery delay (days)", -30, 60, 0)
        late_deliveries = st.number_input("number of late deliveries", 0, 20, 0, step=1)
        avg_freight_value = st.number_input("average freight cost (BRL)", 0.0, 200.0, 20.0, step=5.0)
        avg_product_weight = st.number_input("average product weight (g)", 0.0, 30000.0, 1000.0, step=100.0)

    with col3:
        st.subheader("payment & reviews")
        avg_installments = st.slider("average installments", 1, 24, 1)
        unique_payment_types = st.slider("unique payment types used", 1, 4, 1)
        used_voucher = st.selectbox("used a voucher?", [0, 1], format_func=lambda x: "yes" if x == 1 else "no")
        avg_review_score = st.slider("average review score", 1.0, 5.0, 4.0, step=0.1)
        min_review_score = st.slider("lowest review score given", 1, 5, 3)
        gave_bad_review = st.selectbox("ever gave a bad review (≤2)?", [0, 1], format_func=lambda x: "yes" if x == 1 else "no")

    st.markdown("---")

    if st.button("predict", type="primary"):
        input_data = pd.DataFrame([[
            recency, monetary, avg_delivery_delay, max_delivery_delay,
            late_deliveries, avg_installments, unique_payment_types,
            used_voucher, avg_item_price, max_item_price,
            avg_product_weight, avg_freight_value, avg_review_score,
            min_review_score, gave_bad_review
        ]], columns=FEATURES)

        churn_prob = model.predict_proba(input_data)[0][1]
        churn_pct = churn_prob * 100

        # risk level
        if churn_pct >= 80:
            risk = "🔴 high risk"
        elif churn_pct >= 50:
            risk = "🟠 medium risk"
        else:
            risk = "🟢 low risk"

        col1, col2 = st.columns(2)
        with col1:
            st.metric("churn probability", f"{churn_pct:.1f}%")
            st.markdown(f"**risk level**: {risk}")

        with col2:
            st.subheader("business recommendation")
            if avg_product_weight > 5000 and churn_pct > 80:
                st.warning("this customer bought a large, heavy item — likely a one-off purchase. a discount voucher is unlikely to drive a second purchase. consider recommending complementary accessories instead.")
            elif avg_delivery_delay > 10 and churn_pct > 80:
                st.error("this customer experienced significant delivery delays. prioritize a service recovery message with a meaningful voucher before attempting upsell.")
            elif avg_review_score < 3 and churn_pct > 80:
                st.error("this customer left poor reviews. address their service experience directly before any retention campaign.")
            elif churn_pct < 40:
                st.success("this customer shows strong retention signals. a loyalty reward or early access offer could convert them into a repeat buyer.")
            else:
                st.info("this customer is at moderate churn risk. a targeted email campaign with a personalized product recommendation is a reasonable first step.")

        # shap explanation
        st.markdown("---")
        st.subheader("why did the model make this prediction?")

        shap_values = explainer.shap_values(input_data)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        # top 3 reasons
        feature_impact = pd.DataFrame({
            'feature': FEATURES,
            'shap_value': sv,
            'abs_impact': np.abs(sv)
        }).sort_values('abs_impact', ascending=False)

        top3 = feature_impact.head(3)
        st.markdown("**top 3 factors driving this prediction:**")
        for _, row in top3.iterrows():
            direction = "pushes toward churn" if row['shap_value'] > 0 else "pushes toward retention"
            st.markdown(f"- **{row['feature']}** — {direction} (impact: {row['shap_value']:.3f})")

        # force plot
        st.markdown("**shap force plot — full breakdown of all features:**")
        fig, ax = plt.subplots(figsize=(14, 3))
        shap.force_plot(
            explainer.expected_value,
            sv,
            input_data.iloc[0],
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close()

# ── page 3: model insights ────────────────────────────────────
elif page == "model insights":
    st.title("📊 Model Insights")
    st.markdown("Global analysis of what drives churn across all 55,364 customers.")
    st.markdown("---")

    st.subheader("model performance comparison")
    st.markdown("""
    Three models were trained and compared. Logistic regression served as the baseline.
    LightGBM improved on it significantly, and Optuna hyperparameter tuning pushed the
    final model to an AUC of 0.9991 — meaning the model correctly ranks 99.9% of
    customer pairs by churn likelihood.
    """)
    st.image('../data/model_comparison.png', width=500)

    st.markdown("---")
    st.subheader("roc curve")
    st.markdown("""
    The ROC curve shows the tradeoff between catching churned customers (true positive rate)
    and incorrectly flagging retained ones (false positive rate). A perfect model hugs the
    top-left corner. Our curve is extremely close to perfect, confirming the model is
    highly reliable for this dataset.
    """)
    st.image('../data/roc_curve.png', width=500)

    st.markdown("---")
    st.subheader("global feature importance (shap)")
    st.markdown("""
    This chart shows which features had the biggest average impact on predictions across
    all customers. Delivery delay dominates — both the worst and average delay experienced
    by a customer matter far more than how much they spent or what they reviewed.
    """)
    st.image('../data/shap_importance.png', width=550)

    st.markdown("---")
    st.subheader("shap value distribution")
    st.markdown("""
    This dot plot shows both the magnitude and direction of each feature's impact.
    Red dots represent high feature values, blue dots represent low values.
    Dots to the right push toward churn, dots to the left push toward retention.
    For delivery delay, red dots (high delay) push strongly toward churn — confirming
    late deliveries are the clearest signal that a customer will not return.
    """)
    st.image('../data/shap_summary.png', width=550)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("retained customer")
        st.markdown("""
        This customer received orders 15.5 days early on average.
        That single factor was the dominant reason the model predicted they would return.
        """)
        st.image('../data/shap_retained.png', width=550)

    with col2:
        st.subheader("churned customer")
        st.markdown("""
        This customer bought a very heavy item (9,750g) at a high price point —
        a classic one-off large purchase. The model correctly identified this as
        a one-time buying pattern with little chance of return.
        """)
        st.image('../data/shap_churned.png', width=550)