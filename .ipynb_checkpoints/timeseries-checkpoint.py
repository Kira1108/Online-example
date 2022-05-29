import pandas as pd
import numpy as np
import pmdarima as pm
import matplotlib.pyplot as plt
import streamlit as st
from dataset import get_airline
from plot_timeseries import plot_simple_timeseries, plot_train_test,plot_testset_forecast
import pmdarima as pm

def preprocess_data(df, feature_name):
    df['Observation'] = df[feature_name]
    df['LogObservation'] = np.log(df['Observation'])
    df['Prediction'] = None
    df['UpperBound'] = None
    df['LowerBound'] = None
    return df


def train_sarmia(train, params):
    params.update({'trace':True,'suppress_warnings':True})
    return pm.auto_arima(
    train['Passengers'],
    **params
)
    

def time_series_app():
    df = get_airline()

    st.title("Online Training Demo")

    st.sidebar.markdown("### Select Dataset and Model")
    datasetname = st.sidebar.selectbox("Select Dataset", ("AirlinePassengers",))
    classifier_name = st.sidebar.selectbox("Select Classifier", ("SARIMA",))
    feature_name = st.sidebar.selectbox("Select Feature", df.columns.tolist())
    
    df = preprocess_data(df, feature_name)
    

    st.sidebar.markdown("### Model Parameters")
    
    parameters = {
        "seasonal":st.sidebar.radio("Seasonal",(True, False)),
        "m":st.sidebar.slider("Periods",1,36,12)
    }
    
    st.plotly_chart(plot_simple_timeseries(df))
    Ntest = st.slider("Select forecast steps",1,df.shape[0] - 1,12)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total timesteps: ", df.shape[0])
    col2.metric("Train timesteps: ", df.shape[0] - Ntest)
    col3.metric("Test timesteps: ", Ntest)
    
    start_train = st.button("Start Training")
    if start_train:
        train = df.iloc[:-Ntest]
        test = df.iloc[-Ntest:]
        
        train_caption = st.caption("Training in process...")
        model = train_sarmia(train, parameters)
        train_caption.caption("Training finished")
        
        html_info = model.summary().as_html()
        st.markdown("### Training statistics:")
        st.markdown(html_info, unsafe_allow_html=True)
        
        test_pred, confint = model.predict(n_periods = Ntest, return_conf_int = True)
        test['Prediction'] = test_pred
        test['LowerBound'] = confint[:,0]
        test['UpperBound'] = confint[:,1]
        
        train_pred = model.predict_in_sample(start = 0, end = -1)
        train['Prediction'] = train_pred
        
        st.plotly_chart(plot_testset_forecast(test))
        st.plotly_chart(plot_train_test(train, test,parameters['m']))
        



if __name__ == "__main__":
    time_series_app()