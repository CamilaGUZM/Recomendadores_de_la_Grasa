
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

class UnknownToZero(BaseEstimator, TransformerMixin):
    def __init__(self, column=None):
        self.column=column
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X[self.column]=X[self.column].replace({"M1": 0,"M2": 0,"M3": 0}).astype(float)
        return X
    
class FixRanges(BaseEstimator, TransformerMixin):
    def __init__(self, column=None):
        self.column=column

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X[self.column]=X[self.column].astype(str)
        col1 = []
        col2 = []
        
        for i in X[self.column]:
            col1.append(i[:3])
            col2.append(i[-3:])
        
        X.insert(loc=len(X.columns), column=self.column+" min", value=col1)
        X.insert(loc=len(X.columns), column=self.column+" max", value=col2)
        
        X[self.column+" min"] = pd.to_numeric(X[self.column+" min"])
        X[self.column+" max"] = pd.to_numeric(X[self.column+" max"])
        
        X.drop(columns=[self.column], inplace=True)
        
        return X