import os
from dotenv import load_dotenv
BASE_DIR=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
ENV_PATH=os.path.join(BASE_DIR,".env")
load_dotenv(ENV_PATH)

class Settings:
    POSTGRES_USER= os.getenv("postgres")
    POSTGRES_PASSWORD= os.getenv("Taknevirgi@23")
    POSTGRES_HOST= os.getenv("localhost")
    POSTGRES_PORT= os.getenv("5432")
    POSTGRES_DB= os.getenv("ieeefraud")

    SQLALCHEMY_DATABASE_URL=(
        "postgresql://postgres:Taknevirgi%4023@localhost:5432/ieeefraud"
    )
settings=Settings()
   