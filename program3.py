import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Title
st.title("🛍️ Purchase Prediction App")

# Sidebar inputs
st.sidebar.header("User Input Features")
age = st.sidebar.slider("Age", 18, 70, 30)
salary = st.sidebar.slider("Estimated Salary (in ₹)", 10000, 150000, 50000)

# Sample dataset
data = {
    'Age': [22, 25, 47, 52, 46, 56, 23, 24, 48, 50],
    'EstimatedSalary': [15000, 29000, 48000, 60000, 52000, 79000, 18000, 20000, 50000, 58000],
    'Purchased': [0, 0, 1, 1, 1, 1, 0, 0, 1, 1]
}
df = pd.DataFrame(data)

# Features and target
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_scaled, y)

# Predict user input
user_input = pd.DataFrame([[age, salary]], columns=['Age', 'EstimatedSalary'])
user_scaled = scaler.transform(user_input)
prediction = model.predict(user_scaled)[0]
probability = model.predict_proba(user_scaled)[0][1]

# Output
st.subheader("Prediction Result")
if prediction == 1:
    st.success(f"✅ Likely to Purchase (Confidence: {probability:.2f})")
else:
    st.warning(f"❌ Unlikely to Purchase (Confidence: {probability:.2f})")

# Show dataset
with st.expander("📊 Show Training Data"):
    st.dataframe(df)