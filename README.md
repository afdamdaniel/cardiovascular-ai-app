# Cardiovascular ML + LLM Streamlit App

A small, readable starter project that uses **your existing trusted `.pkl` model** for screening classification and an **optional** LLM to explain the output. It intentionally does not include or invent a trained medical model.

> **Important:** This is a portfolio/research prototype, not an approved medical device. Do not use it for diagnosis, clinical decisions, or reassurance about symptoms. The ML model does the classification, and the LLM only explains the label in general terms.

## 1. Project structure

```text
cardiovascular_ai_app/
├── app.py                   # Streamlit interface
├── inspect_model.py         # Helps inspect your .pkl requirements
├── src/
│   ├── __init__.py          # Package marker (no special code required)
│   ├── model_service.py     # Load the .pkl and predict
│   └── llm_service.py       # Request optional general explanations
├── config/
│   └── features.json        # Form fields: YOU MUST EDIT
├── models/
│   └── cardiovascular_model.pkl   # Add your own trusted model
├── .env.example             # Template for private config
├── requirements.txt
└── .gitignore
```

**No `utils.py` needed yet.** Add a utilities file only when you genuinely have multiple reusable helper functions.

## 2. Install (Windows PowerShell)

Use the Python version and scikit-learn version compatible with the environment you used to TRAIN and SAVE your model. If starting fresh, Python 3.12 is a conservative option, but the training environment is the source of truth.

```powershell
py --list
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

If `py -3.12` is unavailable on Windows, install that version first or use a compatible Python version you already have.

**Make sure the scikit-learn version matches training**. An arbitrary upgrade may make a saved model fail to load. If you had a custom transformer in your training code, you also need to make it importable here.

## 3. Add and inspect YOUR model

1. Put your trusted model at `models/cardiovascular_model.pkl` (or update `MODEL_PATH` in `.env`).
2. Run:

```powershell
python inspect_model.py
```

3. Check `feature_names_in_`, the class labels and (if relevant) pipeline steps.
4. Edit `config/features.json` to match **every** expected input feature, spelling, type, units and encoding. The supplied values are **illustrations, not a real cardio dataset schema**.
5. Set `"verified": true` only when you have checked it.
6. Update `POSITIVE_LABEL` in `.env` to the actual positive/disease class used when you trained the model. The default is `1`, but you must verify it.

Your saved `.pkl` should ideally be the full scikit-learn **Pipeline**, including imputation, encoding and scaling, if the original trained model needed those steps. If you only saved a final classifier and performed scaling separately, you must reproduce that identical preprocessing before prediction. You cannot assume the raw inputs will work.

`feature_names_in_` may be absent for models trained on NumPy arrays. In that case, consult your training code for exact feature order.

## 4. Optional API key

Edit `.env`:

```dotenv
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-6-luna
MODEL_PATH=models/cardiovascular_model.pkl
POSITIVE_LABEL=1
```

- The app **does not need a key to run ML predictions**.
- Only clicking **Generate AI explanation** makes an API call.
- Only the model's binary classification is transmitted; no form measurements or identifiers are passed to the LLM.
- The API call has `store=False`. Review the API provider's data/privacy terms before deploying an app to real users.
- Keep `.env` private. Never commit API keys or publish them in a public GitHub repository.
- API usage can incur charges and requires a separate API account/billing setup.

## 5. Run the web app

```powershell
streamlit run app.py
```

Open the local URL shown in your terminal (usually `http://localhost:8501`).

## 6. Workflow

1. User fills in the form.
2. `app.py` calls `src.model_service.predict(...)`.
3. Your trained model returns its original class, and optionally a class probability estimate if supported.
4. The app shows an explicitly non-diagnostic result. **A model score is not a validated clinical risk percentage.**
5. Optionally, user requests a plain-language explanation. `src.llm_service.generate_explanation(...)` sends only the binary class to the LLM.

## 7. Common errors

- **ModuleNotFoundError: sklearn**: install `scikit-learn` in the *active* Python environment.
- **Feature names do not match**: edit `features.json` so feature names and data types agree with model training.
- **Incorrect predictions**: verify original feature order, encodings, age units, transformed features, and whether preprocessing was saved.
- **Can't unpickle custom transformer**: your custom class/function from training must be importable at the same module path used when saved.
- **Unrecognized/unsupported pickle**: use the correct serialization library, Python and scikit-learn versions.
- **Missing key**: create `.env` using `.env.example`; the prediction still works without LLM.

## 8. Safety and security notes

- Never load a `.pkl` file from an untrusted source. Loading pickle/joblib can execute code.
- Keep health measurements private. This demo intentionally does not save inputs or call the LLM with those values.
- No output from this demo should be used as a diagnosis or basis for medical treatment.
- Before real-world use, validate your model on representative external data, assess bias/calibration and get qualified clinical and privacy review.
