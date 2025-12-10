import pandas as pd
from sqlalchemy import create_engine
from src.config import settings
from src.logger import logging 
from src.exception import CustomException
import sys

class PostgresConnector:
   def __init__(self):
      try:
         logging.info("connecting to database ...")
         self.engine=create_engine(settings.SQLALCHEMY_DATABASE_URL)
         logging.info("successfully connected to database")
      except Exception as e:
          raise CustomException(e,sys)
       

   def fetch_data(self,table:str):
     try:
        logging.info("fetching data from database")
        query=f"SELECT * FROM {table}"
        df=pd.read_sql(query,self.engine)
        logging.info(f"fetching data completed, shape of data is {df.shape}")
        return df
     except Exception as e:
        raise CustomException(e,sys)
