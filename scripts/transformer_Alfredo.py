import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class FillNaNsWithCeros(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Solo llenar con ceros si hay NaNs
        if X.isnull().values.any():
            X = X.fillna(0)
        return X 
