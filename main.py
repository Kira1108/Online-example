from time import time
import streamlit as st
from classification import classificaion_app
from timeseries import time_series_app
from yield_app import analytical_app
from plugin_app import plugin_app
st.sidebar.markdown("### Select ML Problem")

problem = st.sidebar.radio("Choose Problem", ("Classification", "Regression","Analytical", "Plugin"))

if problem == 'Classification':
    classificaion_app()
    
elif problem == 'Regression':
    time_series_app()
    
elif problem == 'Analytical':
    analytical_app()
    
elif problem == 'Plugin':
    plugin_app()
    
    