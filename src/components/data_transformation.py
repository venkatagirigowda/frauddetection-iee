from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.features.customfeatures import CustomFeature
import os
import sys
from src.utils import FrequencyEncoder
import pandas as pd
from src.utils import save_object_preprocessor
import numpy as np

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str=os.path.join('artifacts','preprocessor.pkl')
    Xtrain_data_path=os.path.join('artifacts','X_train.npy')
    Xtest_data_path=os.path.join('artifacts','X_test.npy')
    ytrain_data_path=os.path.join('artifacts','y_train.npy')
    ytest_data_path=os.path.join('artifacts','y_test.npy')

class DataTransformation:
    def __init__(self):
        self.data_transformation=DataTransformationConfig()
    
    def data_transformation_obj(self):
        logging.info("data transformation initiated")
        try:
            train_data=pd.read_csv('artifacts/train.csv')
            test_data=pd.read_csv('artifacts/test.csv')
            nominal_cols = ['ProductCD', 'card4', 'card6']
            nominal_high = ['P_emaildomain', 'R_emaildomain', "id_30", "id_31", "DeviceInfo", "id_33"]
            ordinal_cols = ['M4', 'M2', 'M3', 'M5', 'M6']

            numeric_cols_original = []
            for col in train_data.columns:
             if col not in nominal_cols and \
             col not in nominal_high and \
             col not in ordinal_cols and \
             col != "isFraud" and \
             train_data[col].dtype != "object":
              numeric_cols_original.append(col)

            X_train = train_data.drop(columns=["isFraud"])
            y_train = train_data["isFraud"]
            X_test = test_data.drop(columns=["isFraud"])
            y_test = test_data["isFraud"]

            fe=CustomFeature()
            X_train=fe.fit_transform(X_train)
            X_test=fe.transform(X_test)

            engineered_features = [
                 'Transaction_Hour', 'Transaction_DayOfWeek', 'Transaction_Day',
                 'card1_Count', 'card1_Amt_Mean', 'card1_Amt_Std',
                 'addr1_Count', 'addr1_Amt_Mean', 'addr1_Amt_Std',
                 'Amt_Div_Mean_card1', 'TransactionAmt_Log'
                                     ]
            numeric_cols = numeric_cols_original + engineered_features
            
            #removing the TransactionDT after extracting the feature from it 
            if 'TransactionDT' in X_train.columns:
                 X_train.drop(columns=['TransactionDT'], inplace=True)
                 X_test.drop(columns=['TransactionDT'], inplace=True)
                 numeric_cols.remove('TransactionDT')
            
            #Frequency Encoder is in utils.py

            ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            freq = FrequencyEncoder()
            scaler = StandardScaler()

            preprocessor = ColumnTransformer(
            transformers=[
                ("nominal", ohe, nominal_cols),
                ("ordinal", oe, ordinal_cols),
                ("freq", freq, nominal_high), # nominal_high columns has high cardinality categorical features 
                ("num", scaler, [col for col in numeric_cols if col in X_train.columns]), 
                  ],remainder='drop')
            
            X_train_preprocessed = preprocessor.fit_transform(X_train)
            X_test_preprocessed = preprocessor.transform(X_test)

            logging.info("preprocessing applied/completed")
            
            """def get_feature_names(ct):
             names = []
             for name, trans, cols in ct.transformers_:
                 if name != 'remainder':
                  if hasattr(trans, "get_feature_names_out"):
                   names.extend(trans.get_feature_names_out(cols))
                  else:
                      names.extend(cols)
             return names
            
            train_cols = get_feature_names(preprocessor)

            X_train_transformed = pd.DataFrame(X_train_preprocessed, columns=train_cols)
            X_test_transformed = pd.DataFrame(X_test_preprocessed, columns=train_cols)

            pca = PCA(n_components=0.95, random_state=42)
            X_train_pca = pca.fit_transform(X_train_transformed)
            X_test_pca = pca.transform(X_test_transformed)

            df_augmentation = pd.DataFrame(X_train_transformed.copy())
            df_augmentation["isFraud"] = y_train.values

            #using mhsysgen for augmentation my original augmentation technique

            mh = MHSysGen(method="parallel", ratio=15, minority_class=1)

        
            X_train_aug, y_train_aug = mh.fit_resample(df_augmentation, target="isFraud")
    

            X_aug_pca = pca.transform(X_train_aug)

            save_object_preprocessor(file_path=self.data_transformation.preprocessor_obj_file_path,
                        obj=preprocessing_obj)

            return(X_aug_pca,y_train_aug,X_test_pca,y_test)"""
            logging.info("saving preprocessor obj")
            np.save(self.data_transformation.Xtrain_data_path,X_train_preprocessed)
            np.save(self.data_transformation.Xtest_data_path,X_test_preprocessed)
            np.save(self.data_transformation.ytrain_data_path,y_train)
            np.save(self.data_transformation.ytest_data_path,y_test)

            save_object_preprocessor(file_path=self.data_transformation.preprocessor_obj_file_path,obj=preprocessor)
            logging.info("preprocessor obj saved successfully")

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    initiate_data_transformation=DataTransformation()
    initiate_data_transformation.data_transformation_obj()

