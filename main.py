from time import time
import streamlit as st
from classification import classificaion_app
from timeseries import time_series_app
from analytical import analytical_app

st.sidebar.markdown("### Select ML Problem")

problem = st.sidebar.radio("Choose Problem", ("Classification", "Regression","Plugin"))

if problem == 'Classification':
    classificaion_app()
    
elif problem == 'Regression':
    time_series_app()
    
elif problem == 'Plugin':
    analytical_app()
    
    