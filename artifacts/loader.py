from pathlib import Path
import sys
from src.logger import logging
from src.exception import CustomException
from fastapisetup.artifact_registry import ArtifactRegistry
from src.utils import load_object_pickle, load_object_dill


BASE_DIR = Path(__file__).resolve().parent 


def load_artifacts():
    try:
        logging.info("Loading inference artifacts from disk...")

        model_path = BASE_DIR / "model.pkl"
        preprocessor_path = BASE_DIR / "preprocessor.pkl"
        feature_eng_path = BASE_DIR / "feature_engineering.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Missing preprocessor file: {preprocessor_path}")
        if not feature_eng_path.exists():
            raise FileNotFoundError(f"Missing feature engineering file: {feature_eng_path}")

        ArtifactRegistry.model = load_object_pickle(model_path)
        ArtifactRegistry.preprocessor = load_object_dill(preprocessor_path)
        ArtifactRegistry.feature_engineering = load_object_dill(feature_eng_path)

        logging.info("All inference artifacts loaded successfully.")

    except Exception as e:
        logging.error("Artifact loading failed", exc_info=True)
        raise CustomException(e,sys)