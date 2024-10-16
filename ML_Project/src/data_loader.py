# src/data_loader.py
import pandas as pd
from sklearn.datasets import load_iris

class DataLoader:
    def __init__(self):
        self.data = None

    def load_data(self):
        iris = load_iris()
        data = pd.DataFrame(iris.data, columns=iris.feature_names)
        data['target'] = iris.target
        self.data = data
        return self.data
