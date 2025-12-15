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
from artifacts.loader import load_artifacts
import uvicorn

MAX_ROW = 100
GLOBAL_PREDICTOR = None
MODEL_STATUS = "INITIALIZING"
LOADING_ERROR_DETAIL = None

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global GLOBAL_PREDICTOR, MODEL_STATUS, LOADING_ERROR_DETAIL

    try:
        logging.info("Application startup: loading ML artifacts...")
        load_artifacts()                     # DVC pull + load
        GLOBAL_PREDICTOR = PredictPipeline()
        MODEL_STATUS = "READY"
        logging.info("Service is READY.")
    except Exception as e:
        MODEL_STATUS = "FAILED"
        LOADING_ERROR_DETAIL = str(e)
        logging.critical("Startup failed", exc_info=True)

    yield
    logging.info("Application shutting down.")


app = FastAPI(lifespan=lifespan)


@app.get("/status", response_class=JSONResponse)
async def get_model_status():
    if MODEL_STATUS == "READY":
        return {"status": "READY", "message": "Service operational"}
    return JSONResponse(
        status_code=503,
        content={"status": MODEL_STATUS, "message": LOADING_ERROR_DETAIL}
    )


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_status": MODEL_STATUS,
            "status_detail": LOADING_ERROR_DETAIL or "Ready"
        }
    )


@app.post("/uploadfile/predict", response_class=JSONResponse)
async def upload_predict(file: UploadFile = File(...)):
    if MODEL_STATUS != "READY" or GLOBAL_PREDICTOR is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable. Model not ready."
        )

    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))

        if df.empty:
            raise ValueError("Uploaded CSV is empty.")
        if len(df) > MAX_ROW:
            raise ValueError(f"Max {MAX_ROW} rows allowed.")

        predictions = GLOBAL_PREDICTOR.predict(df)

        return {
            "status": "SUCCESS",
            "predictions": predictions.to_dict("records")
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)