import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from fastapisetup.artifact_registry import ArtifactRegistry
import sys


class PredictPipeline:
    def _init_(self):
        # No loading here by design
        pass

    def predict(self, df: pd.DataFrame):
        try:
            logging.info("Starting stacking prediction pipeline.")

            # ---- Fetch artifacts ----
            pipeline = ArtifactRegistry.model
            fe = ArtifactRegistry.feature_engineering
            pre = ArtifactRegistry.preprocessor

            if pipeline is None or fe is None or pre is None:
                raise RuntimeError("Artifacts not loaded in registry")

            pca = pipeline["pca_transformer"]
            model_xgb = pipeline["base_model_xgb"]
            model_cat = pipeline["base_model_cat"]
            meta_model = pipeline["meta_model_lr"]
            threshold = pipeline["stacking_threshold"]

            # ---- Feature engineering ----
            df_fe = fe.transform(df.copy())

            # ---- Preprocessing ----
            X_scaled = pre.transform(df_fe)

            # ---- PCA ----
            X_pca = pca.transform(X_scaled)

            # ---- Base model predictions ----
            p_xgb = model_xgb.predict_proba(X_pca)[:, 1]
            p_cat = model_cat.predict_proba(X_pca)[:, 1]

            # ---- Stacking ----
            stack_input = np.column_stack([p_xgb, p_cat])

            # ---- Meta model ----
            proba = meta_model.predict_proba(stack_input)[:, 1]
            final_pred = (proba >= threshold).astype(int)

            logging.info("Prediction completed successfully.")

            return pd.DataFrame({
                "fraud_probability": proba,
                "is_fraud_prediction": final_pred
            })

        except Exception as e:
            logging.error("Prediction pipeline failed", exc_info=True)
            raise CustomException(e,sys)