import plotly.graph_objs as go
import numpy as np
import pandas as pd

def plot_simple_timeseries(df, feature_name = None):
    y = df.iloc[:,0]
    
    if feature_name:
        y = df[feature_name]
        
    else:
        feature_name = df.columns[0]
    
    fig = go.Figure([
        go.Scatter(
            name='ObservedData',
            x=df.index,
            y=y,
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        )])
    
    fig.update_layout(
        yaxis_title=feature_name,
        xaxis_title = "Datetime",
        title=f"Time series raw data",
        hovermode="x"
    )
    
    return fig

def plot_time_series_with_bounds(df,data_col = None, upper_col = None, lower_col = None):
    
    if not data_col:
        data_col = df.columns[0]
    if not upper_col:
        upper_col = df.columns[1]
    if not lower_col:
        lower_col = df.columns[2]
        
    fig = go.Figure([
        go.Scatter(
            name='data',
            x=df.index,
            y=df[data_col],
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        go.Scatter(
            name='Upper Bound',
            x=df.index,
            y=df[upper_col],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=df.index,
            y=df[lower_col],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])

    fig.update_layout(
        yaxis_title='Wind speed (m/s)',
        title='Continuous, variable value error bars',
        hovermode="x"
    )
    
    return fig

def plot_testset_forecast(test):
    
    fig = go.Figure(
        [
        # show predictions
        go.Scatter(
        x = test.index,
        y = test.Observation,
        mode = 'lines',
        name = "Observation"), 
        
        # show actual data values
        go.Scatter(
        x = test.index,
        y = test.Prediction,
        mode = 'lines',
        name = "Prediction"), 
        ]
    )

    y_upper = test.UpperBound
    y_lower = test.LowerBound

    fig.add_trace(go.Scatter(
            x=np.concatenate([test.index, test.index[::-1]]),
            y=pd.concat([y_upper, y_lower[::-1]]),
            fill='toself',
            hoveron='points',
            name="Confidence Interval(95%)",
            fillcolor='rgba(68, 68, 68, 0.15)',
            )
    )

    fig.update_layout(
            {
            'title': {"text":"Testset Forecast"},
            'plot_bgcolor':"white"
            })

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(127,127,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(127,127,128,0.2)')
    return fig



def plot_train_test(train, test, seasonal):
    fig = go.Figure([
        go.Scatter(
            x = train.index,
            y = train.Observation,
            name = "Training Observation"
        ),
        go.Scatter(
            x = train.iloc[seasonal + 1:,: ].index,
            y = train.iloc[seasonal + 1:,: ].Prediction,
            name = "Train Prediction",
            line = dict(dash = "dash")
        ),
        go.Scatter(
            x = test.index,
            y = test.Prediction,
            mode = 'lines',
            line = dict(dash = "dash"),
            name = "Test Prediction"), 
        
        go.Scatter(
            x = test.index,
            y = test.Observation,
            mode = 'lines',
            name = "Test Observation")
        
        ]
        
    )
    
    y_upper = test.UpperBound
    y_lower = test.LowerBound

    fig.add_trace(go.Scatter(
            x=np.concatenate([test.index, test.index[::-1]]),
            y=pd.concat([y_upper, y_lower[::-1]]),
            fill='toself',
            hoveron='points',
            name="Confidence Interval(95%)",
            fillcolor='rgba(68, 68, 68, 0.25)',
            line_color = "rgba(0,0,0,0)"
            )
    )

    fig.update_layout(
            {
            'title': {"text":"Prediction Overview"},
            'plot_bgcolor':"white"
            })

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(127,127,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(127,127,128,0.2)')
    return fig