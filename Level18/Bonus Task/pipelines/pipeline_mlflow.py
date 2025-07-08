import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from steps.ingest_data import ingest_data
from steps.train_model import train_model

def run_pipeline():
    X, y = ingest_data()
    model = train_model(X, y)

if __name__ == "__main__":
    run_pipeline()
