import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df, iris

def train_model(df):
    X = df.iloc[:, :-1]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def main():
    st.title("Iris Flower Prediction App")
    df, iris = load_data()
    model = train_model(df)
    
    st.write("### Enter feature values to predict the flower species")
    
    sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.0)
    sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.0)
    petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=4.0)
    petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=1.5)
    
    if st.button("Predict"):
        input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_features)
        predicted_species = iris.target_names[prediction[0]]
        st.success(f"Predicted Species: {predicted_species}")

if __name__ == "__main__":
    main()
