import streamlit as st


from dataset import get_airline
import plotly.graph_objects as go
import pickle

def plugin_app():
    
    st.title("Online Training Demo")
    
    
    df = get_airline()

    Ntest = 12
    train = df.iloc[:-Ntest]
    test = df.iloc[-Ntest:]


    def get_arima():
        with open("arima.pkl",'rb') as f:
            arima = pickle.load(f)
        return arima

    arima = get_arima()
    arima_pred, confint = arima.predict(n_periods = Ntest, return_conf_int = True)


    from nelson_siegel_svensson import NelsonSiegelSvenssonCurve
    import numpy as np

    st.sidebar.markdown("### Model Parameters")
    neilson_parameter_ui = {
                "beta0": st.sidebar.slider("beta1", -1.0, 1000.0, 0.028),
                # "beta1": st.sidebar.slider("beta1", 0.0,1.0, -0.03),
                "beta2": st.sidebar.slider("beta2", -1.0,1.0, -0.04),
                # "beta3": st.sidebar.slider("beta3", -1.0,1.0, 0.0),
                "tau1": st.sidebar.slider("tau1", 0.1, 10.0, 1.1),
                # "tau2": st.sidebar.slider("tau2", 0.1, 10.0, 4.0)
            }
            
    neilson_parameter_ui.update({"beta1":- neilson_parameter_ui['beta0']})
    neilson_parameter_ui.update({"beta3":0.0})
    neilson_parameter_ui.update({'tau2':1.0})

    nelson = NelsonSiegelSvenssonCurve(**neilson_parameter_ui)
    nelson_pred = nelson(np.arange(0, Ntest))

    starting = df.shape[0] - Ntest

    fig = go.Figure([
        go.Scatter(name = "Observed Data", x = np.arange(0, len(df)), y = df['Passengers']),
        go.Scatter(name = 'Model Predict', x = np.arange(starting, starting + Ntest), y = arima_pred),
        go.Scatter(name = 'Plugin', x = np.arange(starting, starting + Ntest), y = nelson_pred + arima_pred),
    ])

    fig.update_layout(title = "Plugin Model Prediction")
    st.plotly_chart(fig)

if __name__ == "__main__":
    plugin_app()
# arima predict -  model predict
# nelson predict - plugin
# beta0 -> beta1