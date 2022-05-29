import streamlit as st
from dataset import get_dataset
from models import get_model
import numpy as np
from parameter_ui import get_parameters
import pandas as pd
from train import train_model

def classificaion_app():
    # ========= Sidebar =========
    st.sidebar.markdown("### Select Dataset and Model")
    datasetname = st.sidebar.selectbox("Select Dataset", ("iris","Boston","Breast Cancer"))
    classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN","SVM","Random Forest"))

    st.sidebar.markdown("### Model Parameters")
    parameters = get_parameters(classifier_name)


    # ========= Data Section =========
    st.title("Online Training Demo")

    st.markdown("Explore different Machine Learning Algorithm")
    st.markdown("""
                Which one is the best
                """)

    st.markdown("### Dataset: " + datasetname)

    X,y = get_dataset(datasetname)
    st.write("Features shape", X.shape)
    st.write("Target shape", y.shape)
    st.write("Number of classes: ", np.unique(y).shape[0])
    features = pd.DataFrame(data=X, columns = [f"feature_{i+1}"for i in range(X.shape[1])])
    target = pd.DataFrame(data=y, columns = ["target"])
    table = pd.concat([features, target], axis=1)
    st.table(table.head(10))

    # ========= Training Section =========
    train_split = st.slider("Train set Proportion", 0.1, 0.95, 0.7)

    start_train = st.button("Start Training")
    if start_train:
        show_text = st.caption("Training in process...")
        modelclass = get_model(classifier_name)
        train_model(modelclass, parameters, X, y, 1-train_split)
        show_text.caption("Training finished")
        
        start_train = st.button("Save Experiment")
    
if __name__ == "__main__":
    classificaion_app()
    
    
    
    