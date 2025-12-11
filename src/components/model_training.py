from src.logger import logging
from src.exception import CustomException
from mh_sys_gen import MHSysGen
from sklearn.decomposition import PCA
import pandas as pd
import os
import sys
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from src.utils import save_object_model
import mlflow
from sklearn.metrics import f1_score,roc_auc_score,confusion_matrix

class ModelTrainingConfig():
    model_obj_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer():
    def __init__(self):
        self.model_training=ModelTrainingConfig()

    def initiate_model_training(self):
        logging.info("model trainer started")
        with mlflow.start_run(run_name="IEEE-Fraud detection") as run:
            try:
                X_train=pd.read_csv('artifacts/X_train.csv')
                X_test=pd.read_csv('artifacts/X_test.csv')
                y_train=pd.read_csv('artifacts/y_train.csv').iloc[:, 0].values.ravel()
                y_test=pd.read_csv('artifacts/y_test.csv').iloc[:, 0].values.ravel()
                
                logging.info("Applying PCA on X_train and X_test")
                PCA_VARIANCE=0.90
                mlflow.log_param("Retained Pca-Variance",PCA_VARIANCE)
                pca = PCA(n_components=PCA_VARIANCE, random_state=42)
                X_train_pca = pca.fit_transform(X_train)
                X_test_pca = pca.transform(X_test)

                logging.info(f"PCA completed. Train PCA shape: {X_train_pca.shape}")
                
                logging.info("MHSysGen augmentation started")
                AUGMENTATION_RATIO=15
                mlflow.log_param("MHSysGen Ratio",AUGMENTATION_RATIO)
                df_aug = pd.DataFrame(X_train.copy())
                df_aug["isFraud"] = y_train

                mh = MHSysGen(method="parallel", ratio=AUGMENTATION_RATIO, minority_class=1)

                X_train_aug, y_train_aug = mh.fit_resample(df_aug, target="isFraud")
                logging.info(f"Augmentation complete. New shape: {X_train_aug.shape}")
                #only applying transformation on X_train_aug so no data leakage
                X_train_aug_pca = pca.transform(X_train_aug)
                
                # we are using same parameters and some modification where we are getting good results
                #hyper parameter tuning was done in notebook
                CAT_PARAMS={
                                'iterations':1500,
                                'depth':10,
                                'learning_rate':0.05,
                                'loss_function':'Logloss',
                                'eval_metric':'F1',
                                'scale_pos_weight':15,
                                'random_seed':42,
                                'verbose':0
                }
                mlflow.log_params({f"catboost_{k}": v for k, v in CAT_PARAMS.items()})
                model_cat = CatBoostClassifier( **CAT_PARAMS)
                logging.info("started training catboost")
                model_cat.fit(X_train_aug_pca, y_train_aug)
                logging.info("catboost training completed")
                
                XGB_PARAMS={
                    'n_estimators':600,
                    'max_depth':18,
                    'min_child_weight':5,
                    'learning_rate':0.1,
                    'subsample':0.8,
                    'colsample_bytree':0.8,
                    'scale_pos_weight':1,
                    'random_state':42,
                    'eval_metric':"logloss" 
                }
                mlflow.log_params({f"xgb_{k}": v for k, v in XGB_PARAMS.items()})
                model_xgb = XGBClassifier( **XGB_PARAMS)
                logging.info("started training xgboost")
                model_xgb.fit(X_train_aug_pca, y_train_aug)
                logging.info("xgboost training completed")


                p_xgb_train=model_xgb.predict_proba(X_train_pca)[:,1]
                p_cat_train=model_cat.predict_proba(X_train_pca)[:,1]
                stack_train=np.column_stack([p_xgb_train,p_cat_train])

                mlflow.log_param("meta_model_type", "LogisticRegression")
                meta_model=LogisticRegression(max_iter=1000)
                logging.info("training meta_model Logistic_regression")
                meta_model.fit(stack_train,y_train)
                logging.info("meta_model training completed")
                p_xgb_test=model_xgb.predict_proba(X_test_pca)[:,1]
                p_cat_test=model_cat.predict_proba(X_test_pca)[:,1]
                stack_test=np.column_stack([p_xgb_test,p_cat_test])
                
                OPTIMAL_THRESHOLD=0.2
                mlflow.log_param("final_threshold", OPTIMAL_THRESHOLD)
                proba=meta_model.predict_proba(stack_test)[:,1]
                final_pred=(proba >=OPTIMAL_THRESHOLD).astype(int)

                rc_score = roc_auc_score(y_test, final_pred)
                f1 = f1_score(y_test, final_pred)
                
                mlflow.log_metric("test_roc_auc_score", rc_score)
                mlflow.log_metric("test_f1_score_at_threshold", f1)

                pipeline = {
                    "pca_transformer": pca,
                    "base_model_xgb": model_xgb,
                    "base_model_cat": model_cat,
                    "meta_model_lr": meta_model,
                    "stacking_threshold": OPTIMAL_THRESHOLD 
                }

    
                
                save_object_model(self.model_training.model_obj_file_path,pipeline)
                logging.info("model.pkl has been saved")
                mlflow.log_artifact(self.model_training.model_obj_file_path)
                logging.info("Complete stacking pipeline logged to MLflow as an artifact.")
                
                # Optional: Log evaluation reports as text artifacts
                mlflow.log_text(classification_report(y_test, final_pred), "classification_report.txt")
                mlflow.log_text(str(confusion_matrix(y_test, final_pred)), "confusion_matrix.txt")

            except Exception as e:
                raise CustomException(e,sys)
        
if __name__=='__main__':
  model_trainer=ModelTrainer()
  model_trainer.initiate_model_training()