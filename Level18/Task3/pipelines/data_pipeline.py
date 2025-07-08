from zenml import pipeline
from steps.load_data import load_digits_data
from steps.clean_data import clean_data

@pipeline
def simple_data_pipeline():
    X, y = load_digits_data()   
    clean_data(X, y)
