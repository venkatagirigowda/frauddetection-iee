#frequency encoding 
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import os
import dill
from src.logger import logging
from src.exception import CustomException
import pickle
import sys
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_maps_={}
    
    def fit(self,X,y=None):
        X=pd.DataFrame(X)
        for col in X.columns:
            freq_map=X[col].value_counts(normalize=True)
            self.freq_maps_[col]=freq_map
        return self
    
    def transform(self,X):
        X=pd.DataFrame(X)
        X_transformed=X.copy()
        for col in X.columns:
            X_transformed[col]=X_transformed[col].map(self.freq_maps_[col]).fillna(0)
        return X_transformed
    

import os
import sys
import dill
import pickle # Need to import pickle for model saving/loading
from src.exception import CustomException

# --- Saving Functions (No change needed, but included for completeness) ---

def save_object_preprocessor(file_path, obj):
    """Saves the preprocessor artifact using dill."""
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, 'wb') as file:
        dill.dump(obj, file)

def save_object_model(file_path, obj):
    """Saves the model artifact using standard pickle."""
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)

# --- Loading Functions (CRITICAL MODIFICATION HERE) ---

def load_object_dill(file_path):
    """Loads an object saved with dill (used for Preprocessor)."""
    try:
        with open(file_path, 'rb') as file:
            obj = dill.load(file)
        return obj
    except Exception as e:
        raise CustomException(f"Dill Load Error for {file_path}: {e}", sys)

def load_object_pickle(file_path):
    """Loads an object saved with standard pickle (used for Model)."""
    try:
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        return obj
    except Exception as e:
        raise CustomException(f"Pickle Load Error for {file_path}: {e}", sys)

# REMOVE the old single `load_object` function!
# def load_object(file_path): ...
 

