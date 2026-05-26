import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# --------------------------------
# Page Config
# --------------------------------
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="centered"
)

st.title("🍷 Wine Quality Prediction using SVR")

# --------------------------------
# Load Dataset
# --------------------------------
try:
    df = pd.read_csv("wine_quality.csv")
except FileNotFoundError:
    st.error("wine_quality.csv file not found")
    st.stop()

# --------------------------------
# Show Dataset
# --------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# --------------------------------
# Features and Target
# --------------------------------
X = df.drop("quality", axis=1)
y = df["quality"]

# --------------------------------
# Feature Scaling
# --------------------------------
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = x_scaler.fit_transform(X)

y_scaled = y_scaler.fit_transform(
    y.values.reshape(-1, 1)
).flatten()

# --------------------------------
# Train Test Split
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_scaled,
    test_size=0.2,
    random_state=42
)

# --------------------------------
# Train Model
# --------------------------------
model = SVR(kernel="rbf")

model.fit(X_train, y_train)

# --------------------------------
# Model Evaluation
# --------------------------------
y_pred_scaled = model.predict(X_test)

y_pred = y_scaler.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
)

y_test_original = y_scaler.inverse_transform(
    y_test.reshape(-1, 1)
)

r2 = r2_score(y_test_original, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))

st.subheader("Model Performance")

st.write(f"R² Score: {r2:.3f}")
st.write(f"RMSE: {rmse:.3f}")

# --------------------------------
# User Input
# --------------------------------
st.subheader("Enter Wine Features")

fixed_acidity = st.number_input("Fixed Acidity", value=7.4)
volatile_acidity = st.number_input("Volatile Acidity", value=0.70)
citric_acid = st.number_input("Citric Acid", value=0.00)
residual_sugar = st.number_input("Residual Sugar", value=1.9)
chlorides = st.number_input("Chlorides", value=0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0)
density = st.number_input("Density", value=0.9978)
pH = st.number_input("pH", value=3.51)
sulphates = st.number_input("Sulphates", value=0.56)
alcohol = st.number_input("Alcohol", value=9.4)

# --------------------------------
# Prediction
# --------------------------------
if st.button("Predict Wine Quality"):

    input_df = pd.DataFrame([{
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }])

    # Scale input
    input_scaled = x_scaler.transform(input_df)

    # Predict
    prediction_scaled = model.predict(input_scaled)

    # Convert back
    prediction = y_scaler.inverse_transform(
        prediction_scaled.reshape(-1, 1)
    )

    st.success(
        f"Predicted Wine Quality: {prediction[0][0]:.2f}"
    )