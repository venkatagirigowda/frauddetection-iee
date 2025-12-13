from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import io
from pathlib import Path
import contextlib 
from src.logger import logging
from src.exception import CustomException
from src.pipeline.predict_pipeline import PredictPipeline
import uvicorn


# Global variables
MAX_ROW = 100 
GLOBAL_PREDICTOR = None
MODEL_STATUS = "INITIALIZING" 
LOADING_ERROR_DETAIL = None  


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):

    global GLOBAL_PREDICTOR, MODEL_STATUS, LOADING_ERROR_DETAIL
    
    try:
        logging.info("Attempting to load ML predictor artifacts...")
        GLOBAL_PREDICTOR = PredictPipeline() 
        MODEL_STATUS = "READY"
        logging.info("ML Predictor successfully loaded. Service is READY.")
    except CustomException as e:
        MODEL_STATUS = "FAILED"
        LOADING_ERROR_DETAIL = f"Artifact loading failed: {str(e)}"
        logging.critical(f"FATAL: Artifact loading failed. Service unavailable. Error: {e}")
    except Exception as e:
        MODEL_STATUS = "FAILED"
        LOADING_ERROR_DETAIL = f"Unexpected loading error: {str(e)}"
        logging.critical(f"FATAL: Unexpected error during startup: {e}")
        
    yield 
    
    logging.info("Application shutting down.")

app = FastAPI(lifespan=lifespan) 


@app.get("/status", response_class=JSONResponse)
async def get_model_status():
    """Returns the current status of the prediction service."""
    if MODEL_STATUS == "READY":
        status_code = 200
        message = "Service is fully operational. All artifacts loaded."
    else:
        status_code = 503
        message = LOADING_ERROR_DETAIL if LOADING_ERROR_DETAIL else "Service failed to initialize. Check server logs."
        
    return JSONResponse(status_code=status_code, content={"status": MODEL_STATUS, "message": message})


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_status": MODEL_STATUS, # Pass status to the template
        "status_detail": LOADING_ERROR_DETAIL if MODEL_STATUS == "FAILED" else "Ready to predict."
    })



@app.post('/uploadfile/predict', response_class=JSONResponse)
async def upload_predict(file: UploadFile = File(...)):
    
    # 1. Check Model Status
    if MODEL_STATUS != "READY" or GLOBAL_PREDICTOR is None:
        raise HTTPException(
            status_code=503, 
            detail=f"Service Unavailable. Artifacts are not ready. Status: {MODEL_STATUS}."
        )

    try:
        content = await file.read()
       
        try:
             df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        except Exception:
             raise ValueError("Could not read file. Ensure it is a valid CSV format.")
        
      
        if len(df) == 0:
            raise ValueError("The uploaded CSV file is empty.")
        if len(df) > MAX_ROW:
            raise ValueError(f"Too many rows. Maximum allowed is {MAX_ROW}, but got {len(df)}.")

       
        predictions_df = GLOBAL_PREDICTOR.predict(df)
        

        if 'TransactionID' in df.columns:
            df_results = df[['TransactionID']].copy().merge(
                predictions_df, left_index=True, right_index=True
            )
        else:
            
            df_results = predictions_df
        
        logging.info(f"Predictions generated for {len(df_results)} records.")
        
    
        response_data = df_results.rename(
            columns={'fraud_probability': 'probability score', 'is_fraud_prediction': 'prediction'}
        ).to_dict('records')
        
        return {"status": "SUCCESS", "message": "Predictions generated.", "predictions": response_data}

    except ValueError as ve:
        logging.warning(f"Client request validation error: {ve}")
        raise HTTPException(status_code=400, detail=f"Input Error: {ve}")
        
    except Exception as e:
        error_type = type(e).__name__
        logging.error(f"Processing failed: {error_type} - {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Processing Error: {error_type}. See server logs for details.")
    
#running on 8000 port mlflow already running on 5000 port
if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)