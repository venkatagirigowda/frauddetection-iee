#frequency encoding 
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_maps={}
    
    def fit(self,X,y=None):
        X=pd.DataFrame(X)
        for col in X.columns:
            freq_map=X[col].value_counts(normalize=True)
            self.freq_maps[col]=freq_map
        return self
    
    def transform(self,X):
        X=pd.DataFrame(X)
        X_transformed=X.copy()
        for col in X.columns:
            X_transformed[col]=X_transformed[col].map(self.freq_maps[col]).fillna(0)
        return X_transformed