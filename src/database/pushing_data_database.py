import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sqlalchemy import create_engine
import sys
from src.config import settings

DATABASEURL=settings.SQLALCHEMY_DATABASE_URL
engine=create_engine(DATABASEURL)

df=pd.read_csv("notebook/data/train_cleaned.csv")
try:
 logging.info("data transfer initiated")
 df.to_sql(
    name="transactions",
    con=engine,
    if_exists='replace',
    index=False)
 logging.info("data transfer completed")
except Exception as e:
 raise CustomException(e,sys)