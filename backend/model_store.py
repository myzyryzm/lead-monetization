"""
Train-once + persist the calibrated model.

The engine retrains from pairs.csv on every process start (fine for the demo
scripts, too slow to do per web request). This module trains once, saves the
fitted pipeline to model.pkl with joblib, and loads it thereafter. If the
pickle can't be loaded (e.g. a scikit-learn version mismatch), it retrains and
overwrites rather than crashing.
"""

import os
import joblib

from recommend import train_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
PAIRS_PATH = os.path.join(BASE_DIR, "pairs.csv")

_model = None


def _train_and_save():
    model = train_model(PAIRS_PATH)
    joblib.dump(model, MODEL_PATH)
    return model


def get_model():
    """Return the cached model, loading from disk or training+persisting once."""
    global _model
    if _model is not None:
        return _model

    if os.path.exists(MODEL_PATH):
        try:
            _model = joblib.load(MODEL_PATH)
            return _model
        except Exception as e:  # version skew / corrupt pickle -> retrain
            print(f"[model_store] failed to load {MODEL_PATH} "
                  f"({type(e).__name__}: {e}); retraining from scratch")

    print("[model_store] training model (one-time)...")
    _model = _train_and_save()
    print(f"[model_store] saved model to {MODEL_PATH}")
    return _model


if __name__ == "__main__":
    get_model()
    print("model ready")
