import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

@st.cache_data
def load_data():
    return pd.read_csv("data/winequality.csv")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

def main():
    st.set_page_config(page_title="Wine Quality Predictor", layout="wide")
    st.title("🍷 Wine Quality Predictor")
    st.write("Predict wine quality from physicochemical properties.")

    section = st.sidebar.radio("Go to", ["Overview", "Visualizations", "Model", "Predict"])

    df = load_data()
    model = load_model()

    if section == "Overview":
        st.subheader("Dataset Overview")
        st.write(df.head())
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    elif section == "Visualizations":
        st.subheader("Data Visualizations")
        col = st.selectbox("Feature for histogram", df.columns[:-1])
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig)

        x_feat = st.selectbox("X-axis", df.columns[:-1])
        y_feat = st.selectbox("Y-axis", df.columns[:-1])
        fig2 = px.scatter(df, x=x_feat, y=y_feat, color="quality")
        st.plotly_chart(fig2)

    elif section == "Model":
        st.subheader("Model Performance")
        X = df.drop('quality', axis=1)
        y = df['quality']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        st.metric("Mean Absolute Error", round(mae, 3))

    elif section == "Predict":
        st.subheader("Make a Prediction")
        inputs = {}
        for col in df.columns[:-1]:
            val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
            inputs[col] = val
        if st.button("Predict Quality"):
            X_new = np.array([list(inputs.values())])
            pred = model.predict(X_new)[0]
            st.success(f"Predicted wine quality: {pred:.2f}")

if __name__ == "__main__":
    main()
