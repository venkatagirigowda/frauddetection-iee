from src.database.fetching_data_database import PostgresConnector
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import os
import sys
from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
   train_data_path=os.path.join('artifacts','train.csv')
   test_data_path=os.path.join('artifacts','test.csv')

class DataIngestion:
    def __init__(self):
      self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("data ingestion initiated")
        try:
            connector=PostgresConnector()
            df=connector.fetch_data(table="transactions")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            ##Time based split
            df.sort_values(by="TransactionDT")
            split_index=int(0.8*len(df))
            train_df = df.iloc[:split_index]
            test_df = df.iloc[split_index:]

            train_set=train_df.drop(columns=['TransactionID'])
            test_set=test_df.drop(columns=['TransactionID'])

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True )
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("data ingestion complete")

        except Exception as e:
            raise CustomException(e,sys)

if __name__=='__main__':
    data_ingestion_obj=DataIngestion()
    data_ingestion_obj.initiate_data_ingestion()
    


   