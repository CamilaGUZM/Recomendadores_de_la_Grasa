
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class CheckColumnNames(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        data=X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.global_names=data.columns
        return self
    
    def transform(self, X):
        data=X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        data.columns = self.global_names
        return data 
