from sklearn import datasets
import pandas as pd
import numpy as np

def get_dataset(datasetname):
    
    available_datasets = {
        "iris":datasets.load_iris(),
        "Boston":datasets.load_boston(),
        "Breast Cancer":datasets.load_breast_cancer()
    }
    
    data = available_datasets.get(datasetname, None)
    if datasetname == 'Boston':
        return data.data, (data.target > 30).astype(int)
    return data.data, data.target


def get_airline():
    df = pd.read_csv("airline_passengers.csv", index_col = "Month", parse_dates = True)
    df['LogPassengers'] = np.log(df['Passengers'])
    return df
    
