"""Optional: inspect model input names and class labels before writing features.json."""
import os
from pathlib import Path
from dotenv import load_dotenv
from src.model_service import load_model

root = Path(__file__).resolve().parent
load_dotenv(root / '.env')
path = root / os.getenv('MODEL_PATH', 'models/cardiovascular_model.pkl')
model = load_model(str(path))
print('Model:', type(model).__name__)
print('feature_names_in_:', getattr(model, 'feature_names_in_', 'Not recorded by this model'))
print('n_features_in_:', getattr(model, 'n_features_in_', 'Not recorded by this model'))
print('classes_:', getattr(model, 'classes_', 'Not recorded by this model'))
print('Pipeline steps:', getattr(model, 'named_steps', 'This is not a Pipeline'))
