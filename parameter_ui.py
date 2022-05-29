import streamlit as st

def knn_parameters():
    return dict(
        n_neighbors = st.sidebar.slider("Select n_neighbors", 1, 15, 3),
        weights = st.sidebar.radio("Select weights", ("uniform","distance")),
        algorithm = st.sidebar.radio("Select algorithm", ("auto","ball_tree","kd_tree","brute"))
    )
    
def svm_parameters():
    """return a dictionary of parameter required by svm classifier which received from streamlit"""
    return dict(
        C = st.sidebar.slider("Select C", 0.01, 10.0, 1.0),
        kernel = st.sidebar.radio("Select kernel", ("linear","poly","rbf","sigmoid")),
        degree = st.sidebar.slider("Select degree", 1, 15, 3),
        gamma = st.sidebar.radio("Select gamma", ("scale","auto"))
    )

def random_forest_parameters():
    """a dictionary of random forest parameters"""
    
    return dict(
        n_estimators = st.sidebar.slider("Select n_estimators", 1, 100, 10),
        criterion = st.sidebar.radio("Select criterion", ("gini","entropy","log_loss")),
        max_depth = st.sidebar.slider("Select max_depath", 1, 15, 3),
        min_samples_split = st.sidebar.slider("Select min_sample_split", 2, 15, 2),
        min_samples_leaf = st.sidebar.slider("Select min_samples_leaf", 1, 30, 10),
        min_weight_fraction_leaf = st.sidebar.slider("Select min_weight_fraction_leaf", 0.0, 0.5, 0.0),
        max_features = st.sidebar.radio("Select max_features", ("sqrt","log2")),
    ) 

def get_parameters(model_name):   
    parameter_ui = {
        "KNN":knn_parameters,
        "SVM":svm_parameters,
        "Random Forest":random_forest_parameters
    }
    
    return parameter_ui.get(model_name)()