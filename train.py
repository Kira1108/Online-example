from sklearn.model_selection import train_test_split
import streamlit as st

def train_model(classifier,params, X, y, test_size=0.2):
    """
    Train a model with given parameters
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model = classifier(**params)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    col1, col2 = st.columns(2)
    col1.metric("Train Accuracy", train_score)
    col2.metric("Test Accuracy", test_score)
    return model