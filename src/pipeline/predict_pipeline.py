import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from fastapisetup.app import ArtifactRegistry


class PredictPipeline:
    def __init__(self):
        # ❌ NO FILE LOADING HERE
        pass

    def predict(self, df: pd.DataFrame):
        try:
            logging.info("Starting stacking prediction.")

            # Fetch in-memory artifacts
            fe = ArtifactRegistry.feature_engineer
            pre = ArtifactRegistry.preprocessor
            pca = ArtifactRegistry.pca_transformer
            xgb = ArtifactRegistry.base_model_xgb
            cat = ArtifactRegistry.base_model_cat
            meta = ArtifactRegistry.meta_model_lr
            threshold = ArtifactRegistry.optimal_threshold

            if any(v is None for v in [fe, pre, pca, xgb, cat, meta]):
                raise RuntimeError("Artifacts not loaded")

            # Pipeline
            df_fe = fe.transform(df.copy())
            scaled = pre.transform(df_fe)
            X_pca = pca.transform(scaled)

            p_xgb = xgb.predict_proba(X_pca)[:, 1]
            p_cat = cat.predict_proba(X_pca)[:, 1]

            stack_input = np.column_stack([p_xgb, p_cat])
            proba = meta.predict_proba(stack_input)[:, 1]

            final_pred = (proba >= threshold).astype(int)

            return pd.DataFrame({
                "fraud_probability": proba,
                "is_fraud_prediction": final_pred
            })

        except Exception as e:
            raise CustomException(e)