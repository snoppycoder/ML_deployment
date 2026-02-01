import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# LOAD MODEL + SCALER
# -----------------------------
model = joblib.load("models/xgb_fraud_best.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("E-Commerce Fraud Detection System")
st.markdown("Enter transaction details to check fraud risk.")

# -----------------------------
# USER INPUTS
# -----------------------------
purchase_value = st.number_input("Purchase Value", min_value=0.0)
age = st.number_input("Customer Age", min_value=0)
time_since_signup = st.number_input("Time Since Signup (seconds)", min_value=0)

hour_of_day = st.slider("Hour of Day", 0, 23, 12)
day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 0)

total_transactions = st.number_input("Total Transactions", min_value=0)
transactions_last_24h = st.number_input("Transactions Last 24h", min_value=0)
transactions_last_7d = st.number_input("Transactions Last 7 Days", min_value=0)
transactions_last_30d = st.number_input("Transactions Last 30 Days", min_value=0)

velocity_last_24h = st.number_input("Velocity Last 24h", min_value=0.0)
avg_purchase_value = st.number_input("Average Purchase Value", min_value=0.0)

source = st.selectbox("Traffic Source", ["Direct", "SEO", "Other"])
browser = st.selectbox("Browser", ["FireFox", "IE", "Opera", "Safari", "Other"])
sex = st.selectbox("Sex", ["M", "F"])

# -----------------------------
# FEATURE TEMPLATE (CRITICAL)
# -----------------------------
FEATURE_COLUMNS = [
    'purchase_value', 'age', 'hour_of_day', 'day_of_week',
    'time_since_signup', 'total_transactions',
    'transactions_last_24h', 'transactions_last_7d',
    'transactions_last_30d', 'velocity_last_24h',
    'avg_purchase_value', 'purchase_value_deviation',
    'source_Direct', 'source_SEO',
    'browser_FireFox', 'browser_IE',
    'browser_Opera', 'browser_Safari',
    'sex_M'
]

X = pd.DataFrame(0, index=[0], columns=FEATURE_COLUMNS)

# -----------------------------
# NUMERICAL FEATURES
# -----------------------------
X['purchase_value'] = purchase_value
X['age'] = age
X['hour_of_day'] = hour_of_day
X['day_of_week'] = day_of_week
X['time_since_signup'] = time_since_signup
X['total_transactions'] = total_transactions
X['transactions_last_24h'] = transactions_last_24h
X['transactions_last_7d'] = transactions_last_7d
X['transactions_last_30d'] = transactions_last_30d
X['velocity_last_24h'] = velocity_last_24h
X['avg_purchase_value'] = avg_purchase_value

# Engineered feature
X['purchase_value_deviation'] = purchase_value - avg_purchase_value

# -----------------------------
# ONE-HOT ENCODING
# -----------------------------
if source == "Direct":
    X['source_Direct'] = 1
elif source == "SEO":
    X['source_SEO'] = 1

browser_map = {
    "FireFox": "browser_FireFox",
    "IE": "browser_IE",
    "Opera": "browser_Opera",
    "Safari": "browser_Safari"
}

if browser in browser_map:
    X[browser_map[browser]] = 1

if sex == "M":
    X['sex_M'] = 1

# -----------------------------
# SCALING (CRITICAL)
# -----------------------------
NUMERIC_COLUMNS = [
    'purchase_value', 'age', 'hour_of_day', 'day_of_week',
    'time_since_signup', 'total_transactions',
    'transactions_last_24h', 'transactions_last_7d',
    'transactions_last_30d', 'velocity_last_24h',
    'avg_purchase_value', 'purchase_value_deviation'
]

X[NUMERIC_COLUMNS] = scaler.transform(X[NUMERIC_COLUMNS])

st.subheader("Model Input (After Scaling)")
st.dataframe(X)


# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Check Fraud"):
    prob = model.predict_proba(X)[0][1]

    if prob > 0.587:
        st.error(f"🚨 Fraud Detected\n\nProbability: {prob:.2f}")
    else:
        st.success(f"✅ Legitimate Transaction\n\nProbability: {prob:.2f}")
