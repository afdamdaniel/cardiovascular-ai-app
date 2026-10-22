"""Load the already-trained model and make predictions."""
from functools import lru_cache
from pathlib import Path
import joblib
import pandas as pd
import numpy as np


@lru_cache(maxsize=1)
def load_model(path: str):
    """Load a TRUSTED model once per Python process.

    joblib can read commonly saved scikit-learn .pkl models, including most
    standard pickle files. Never load untrusted .pkl/.joblib files.
    """
    return joblib.load(Path(path))


def predict(model, features: dict, positive_label: str = '1') -> dict:
    """Return model output. Preserve the original model as the sole predictor."""
    # Mirror cardiovascular_prediction.ipynb, cell 32.
    inputs = features.copy()
    if 'age' in inputs:
        if inputs['age'] <= 0:
            raise ValueError('Age must be positive.')
        inputs['age'] = float(np.log(inputs['age']) + 1)
    if 'ap_hi' in inputs and 'ap_lo' in inputs:
        if inputs['ap_hi'] <= inputs['ap_lo']:
            raise ValueError('Systolic pressure must be higher than diastolic pressure.')
    frame = pd.DataFrame([inputs])

    expected = getattr(model, 'feature_names_in_', None)
    if expected is not None:
        expected = list(expected)
        if set(frame.columns) != set(expected) or len(frame.columns) != len(expected):
            missing = [name for name in expected if name not in frame.columns]
            extra = [name for name in frame.columns if name not in expected]
            raise ValueError(f'Feature mismatch. Missing: {missing}; unexpected: {extra}')
        frame = frame[expected]  # model's original training column order

    predicted_class = model.predict(frame)[0]
    is_flagged = str(predicted_class).strip().lower() == str(positive_label).strip().lower()

    positive_class_score = None
    if hasattr(model, 'predict_proba') and hasattr(model, 'classes_'):
        classes = list(model.classes_)
        idx = next((i for i, x in enumerate(classes)
                    if str(x).strip().lower() == str(positive_label).strip().lower()), None)
        if idx is not None:
            positive_class_score = float(model.predict_proba(frame)[0][idx])

    return {
        'predicted_label': str(predicted_class),
        'flagged': is_flagged,
        'positive_class_score': positive_class_score,
    }
