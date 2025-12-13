from src.logger import logging
from src.exception import CustomException
import pandas as pd
import sys
import os
import numpy as np # <-- REQUIRED FOR STACKING LOGIC
from src.utils import load_object_dill, load_object_pickle
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA # <-- REQUIRED FOR PCA TYPE CHECK

class PredictPipeline:
    def __init__(self):
        try:
            # Note: Removed self.model_path and replaced it with individual component paths
            self.model_path = os.path.join('artifacts', 'model.pkl') # This holds the stacking pipeline dictionary
            self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            self.featureengineering_path = os.path.join('artifacts', 'feature_engineering.pkl') # Corrected artifact name 'feature_engineer.pkl'

            # 1. Load the two independent transformers
            self.preprocessor = load_object_dill(file_path=self.preprocessor_path)
            self.feature_engineer = load_object_dill(file_path=self.featureengineering_path)

            # 2. Load the Stacking Pipeline Dictionary
            stacking_pipeline_dict = load_object_pickle(file_path=self.model_path) 
            
            # --- MODIFICATION START: Unpack the Stacking Components ---
            self.pca_transformer = stacking_pipeline_dict.get('pca_transformer')
            self.base_model_xgb = stacking_pipeline_dict.get('base_model_xgb')
            self.base_model_cat = stacking_pipeline_dict.get('base_model_cat')
            self.meta_model_lr = stacking_pipeline_dict.get('meta_model_lr')
            self.optimal_threshold = stacking_pipeline_dict.get('stacking_threshold')
            
            # Basic validation check for loaded components
            if not all([self.pca_transformer, self.base_model_xgb, self.meta_model_lr, self.optimal_threshold is not None]):
                 raise ValueError("Stacking pipeline components missing from model.pkl artifact.")

            # --- MODIFICATION END ---
            
            logging.info("Preprocessor, Feature Engineer, and Stacking Pipeline components loaded.")

            if not isinstance(self.preprocessor, TransformerMixin):
                raise TypeError(
                    f"Loaded object for preprocessor is of type {type(self.preprocessor).__name__}, "
                    f"but expected a Scikit-learn Transformer object with a .transform() method."
                )

        except Exception as e:
            # This catch handles loading errors, including missing files or TypeErrors
            raise CustomException(f"Error loading ML artifacts: {e}", sys)
        # Note: Removed the duplicate except block at the end of __init__

    def predict(self, df:pd.DataFrame):
        try:
            logging.info("Starting sequential stacking prediction on user data.")
            
            # 1. Apply Feature Engineering
            df_fe = self.feature_engineer.transform(df.copy())
            
            # 2. Apply Column Transformation
            data_scaled = self.preprocessor.transform(df_fe)
            
            # 3. Apply PCA
            # preprocessor.transform outputs a numpy array, suitable for PCA
            X_pca = self.pca_transformer.transform(data_scaled)
            logging.info("Data processed through Feature Engineering, Preprocessor, and PCA.")

            # 4. Generate Predictions from Base Models
            # We must call predict_proba on the individual base models
            p_xgb = self.base_model_xgb.predict_proba(X_pca)[:, 1]
            p_cat = self.base_model_cat.predict_proba(X_pca)[:, 1]
            
            # 5. Stack probabilities and run Meta Model
            stack_input = np.column_stack([p_xgb, p_cat])
            
            # The final probability is the output of the meta-model
            proba = self.meta_model_lr.predict_proba(stack_input)[:, 1]
            logging.info("Stacking meta-model predicted raw probabilities.")

            # 6. Apply Optimal Threshold for Final Binary Prediction
            final_pred = (proba >= self.optimal_threshold).astype(int)

            prediction_df=pd.DataFrame({
                'fraud_probability': proba,
                'is_fraud_prediction': final_pred
            })
            
            logging.info("Prediction complete and returning prediction dataframe.")
            return prediction_df
        except Exception as e:
            raise CustomException(e,sys)