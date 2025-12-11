from sklearn.base import BaseEstimator,TransformerMixin
import numpy as np


class CustomFeature(BaseEstimator,TransformerMixin):
    def __init__(self,time_col='TransactionDT',amt_col='TransactionAmt'):
        self.time_col = time_col
        self.amt_col = amt_col
        self.agg_features = {} # Stores aggregation maps during fit
    
    def fit(self, X, y=None):
        X_copy = X.copy()
        X_copy['isFraud'] = y # Temporarily add target for aggregation/target encoding
        
        # 1. Base Aggregation Keys
        self.base_agg_keys = ['card1', 'addr1']
        
        # 2. Prepare Aggregation Maps
        for col in self.base_agg_keys:
            # Frequency Map (for velocity/cardinality)
            self.agg_features[f'{col}_Count_Map'] = X_copy[col].value_counts()
            
            # Amount Mean Map
            self.agg_features[f'{col}_Amt_Mean_Map'] = X_copy.groupby(col)[self.amt_col].mean()
            
            # Amount Std Map
            self.agg_features[f'{col}_Amt_Std_Map'] = X_copy.groupby(col)[self.amt_col].std().fillna(1.0)
            
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        # --- Time-Based Features (Requires TransactionDT) ---
        if self.time_col in X_copy.columns:
            # Hour of Day (0-23)
            X_copy['Transaction_Hour'] = (X_copy[self.time_col] // 3600) % 24
            # Day of Week (0-6)
            X_copy['Transaction_DayOfWeek'] = (X_copy[self.time_col] // (3600 * 24)) % 7
            X_copy['Transaction_Day'] = X_copy[self.time_col] // (3600 * 24)

        # --- Aggregation and Ratio Features ---
        for col in self.base_agg_keys:
            # Frequency Feature
            count_map = self.agg_features[f'{col}_Count_Map']
            X_copy[f'{col}_Count'] = X_copy[col].map(count_map).fillna(0)
            
            # Amount Mean Feature
            mean_map = self.agg_features[f'{col}_Amt_Mean_Map']
            X_copy[f'{col}_Amt_Mean'] = X_copy[col].map(mean_map).fillna(X_copy[self.amt_col].mean())
            
            # Amount Std Feature
            std_map = self.agg_features[f'{col}_Amt_Std_Map']
            X_copy[f'{col}_Amt_Std'] = X_copy[col].map(std_map).fillna(1.0)
        
        # Amount-to-Mean Ratio (highly predictive)
        X_copy['Amt_Div_Mean_card1'] = X_copy[self.amt_col] / X_copy['card1_Amt_Mean']
        
        # Log Transform
        X_copy['TransactionAmt_Log'] = np.log1p(X_copy[self.amt_col])
        
        return X_copy