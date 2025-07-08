import pandas as pd
from sklearn.datasets import load_iris

def ingest_data():
    data = load_iris(as_frame=True)
    df = data.frame
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y
