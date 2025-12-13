from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import io
from pathlib import Path
import contextlib
import os
import uvicorn

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object_dill, load_object_pickle
from src.pipeline.predict_pipeline import PredictPipeline


# ---------------- GLOBAL STATE ----------------
MAX_ROW = 100
GLOBAL_PREDICTOR = None
MODEL_STATUS = "INITIALIZING"
LOADING_ERROR_DETAIL = None

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ---------------- ARTIFACT REGISTRY ----------------
class ArtifactRegistry:
    preprocessor = None
    feature_engineer = None
    pca_transformer = None
    base_model_xgb = None
    base_model_cat = None
    meta_model_lr = None
    optimal_threshold = None


# ---------------- LIFESPAN ----------------
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):

    global GLOBAL_PREDICTOR, MODEL_STATUS, LOADING_ERROR_DETAIL

    try:
        logging.info("Attempting to load ML artifacts...")

        artifacts_dir = "artifacts"

        # Load transformers
        ArtifactRegistry.preprocessor = load_object_dill(
            os.path.join(artifacts_dir, "preprocessor.pkl")
        )
        ArtifactRegistry.feature_engineer = load_object_dill(
            os.path.join(artifacts_dir, "feature_engineering.pkl")
        )

        # Load stacking dictionary
        stacking_dict = load_object_pickle(
            os.path.join(artifacts_dir, "model.pkl")
        )

        ArtifactRegistry.pca_transformer = stacking_dict["pca_transformer"]
        ArtifactRegistry.base_model_xgb = stacking_dict["base_model_xgb"]
        ArtifactRegistry.base_model_cat = stacking_dict["base_model_cat"]
        ArtifactRegistry.meta_model_lr = stacking_dict["meta_model_lr"]
        ArtifactRegistry.optimal_threshold = stacking_dict["stacking_threshold"]

        GLOBAL_PREDICTOR = PredictPipeline()
        MODEL_STATUS = "READY"

        logging.info("✅ All artifacts loaded. Service READY.")

    except Exception as e:
        MODEL_STATUS = "FAILED"
        LOADING_ERROR_DETAIL = str(e)
        logging.critical(f"❌ Startup failed: {e}", exc_info=True)

    yield
    logging.info("Application shutting down.")


app = FastAPI(lifespan=lifespan)


# ---------------- ROUTES ----------------
@app.get("/status", response_class=JSONResponse)
async def get_model_status():
    if MODEL_STATUS == "READY":
        return {"status": MODEL_STATUS, "message": "Service operational"}
    return JSONResponse(
        status_code=503,
        content={"status": MODEL_STATUS, "message": LOADING_ERROR_DETAIL}
    )


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "model_status": MODEL_STATUS}
    )


@app.post("/uploadfile/predict", response_class=JSONResponse)
async def upload_predict(file: UploadFile = File(...)):

    if MODEL_STATUS != "READY" or GLOBAL_PREDICTOR is None:
        raise HTTPException(
            status_code=503,
            detail="Service Unavailable. Artifacts not ready."
        )

    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))

    if len(df) == 0 or len(df) > MAX_ROW:
        raise HTTPException(status_code=400, detail="Invalid input size")

    predictions_df = GLOBAL_PREDICTOR.predict(df)

    return {
        "status": "SUCCESS",
        "predictions": predictions_df.to_dict("records")
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0",port=8000)