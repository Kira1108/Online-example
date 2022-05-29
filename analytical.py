import streamlit as st
import plotly.graph_objects as go
import pandas as pd


from nelson_siegel_svensson import NelsonSiegelSvenssonCurve
from nelson_siegel_svensson.calibrate import calibrate_ns_ols
import numpy as np

from dataset import get_airline

def ols_ui(t, y):
    
    st.sidebar.slider("tau0",0.01,10.0,1.0)
    
    periods = st.slider("Periods Extroploated",1,100,30)
    curve, status = calibrate_ns_ols(t, y, tau0 = 1.0)
    assert status.success
    
    in_sample_t = np.linspace(0,t[-1],200)
    in_sample_pred = curve(in_sample_t)
    out_of_sample_t = np.linspace(t[-1], t[-1] + periods, 200)
    out_of_sample_pred = curve(out_of_sample_t)
    
    
    fig = go.Figure(
        [go.Scatter(name = "Observed Data", x = t, y = y),
         go.Scatter(name = "Fitted Data", x = in_sample_t, y = in_sample_pred),
         go.Scatter(name = "Extroploate", x = out_of_sample_t, y = out_of_sample_pred)
         ]
    )
    
    fig.update_layout(
        title = "Least Square fitted Nelson-Siegel-Svensson curve",
        yaxis_title="Y",
        xaxis_title = "X",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig)
    
    st.markdown("Fitted Parameters")
    
    fitting_df = pd.DataFrame(zip(['beta0','beta1','beta2','tau'],
                     [curve.beta0, curve.beta1, curve.beta2, curve.tau])
                 , columns = ['Parameter', 'Value'])
    
    st.table(fitting_df)
    
def maual_ui(t, y):
    neilson_parameter_ui = {
            "beta0": st.sidebar.slider("beta0", -1.0, 1000.0, 0.028),
            # "beta1": st.sidebar.slider("beta1", 0.0,1.0, -0.03),
            "beta2": st.sidebar.slider("beta2", -1.0,1.0, -0.04),
            # "beta3": st.sidebar.slider("beta3", -1.0,1.0, 0.0),
            "tau1": st.sidebar.slider("tau1", 0.1, 10.0, 1.1),
            # "tau2": st.sidebar.slider("tau2", 0.1, 10.0, 4.0)
        }
        
    neilson_parameter_ui.update({"beta1":- neilson_parameter_ui['beta0']})
    neilson_parameter_ui.update({"beta3":0.0})
    neilson_parameter_ui.update({'tau2':1.0})

    model = NelsonSiegelSvenssonCurve(**neilson_parameter_ui)
    predt = np.linspace(t[-1] + 1,t[-1] + 21,200)

    # fig = go.Figure(
    #     [go.Scatter(name = "Nelsone", x = t, y = model(t)),
    #         go.Scatter(name = "Observed Data", x = t, y = y),
    #         go.Scatter(name = "Extroploate", x = predt, y = model(predt))
    #         ]
    # )
    fig = go.Figure(
        [go.Scatter(name = "Observed Data", x = t, y = y),
         go.Scatter(name = 'Nelson-Siegel-Svensson', x = predt, y = model(np.arange(1, len(predt) + 1)))
         ]
    )

    fig.update_layout(
        title = "Nelson-Siegel-Svensson curve",
        yaxis_title="Y",
        xaxis_title = "X",
        hovermode="x unified"
    )
    st.plotly_chart(fig)
    
    
def choose_dataset(dtype = "simple"):
    if dtype == 'simple':
        t = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
        y = np.array([0.01, 0.011, 0.013, 0.016, 0.019, 0.021, 0.026, 0.03, 0.035, 0.037, 0.038, 0.04])
        return t,y
    elif dtype == 'airline':
        data = get_airline()
        t = np.arange(0, len(data))
        y = data['Passengers']
        return t,y 
    y=np.array([-0.005683497, -0.006091677, -0.006167227, -0.006020827,
                -0.005731884, -0.003626564, -0.001838833, -0.000665441, 0.000556985])
    t =np.array([1,2,3,4,5,10,15,20,30])
    return t,y

    

def analytical_app():
    
    st.sidebar.markdown("### Choose Dataset")
    dtype = st.sidebar.radio("Dataset", ("simple", "complex", "airline"))

    
    st.title("Nelson-Siegel-Svensson curve")
    
    t,y = choose_dataset(dtype)
    
    st.sidebar.markdown("### Choose fitting Method")
    method = st.sidebar.radio("Choose fitting Method",("OLS","Manual"))
    
    st.sidebar.markdown("### Model Parameters")
    if method == "OLS":
        ols_ui(t,y)
    else:
        maual_ui(t, y)
       
    
    
if __name__ == "__main__":
    analytical_app()










