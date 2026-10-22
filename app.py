"""Streamlit UI. Keep model inference and LLM calls in src/ modules."""
from pathlib import Path
import json
import os

from dotenv import load_dotenv
import streamlit as st

from src.model_service import load_model, predict
from src.llm_service import generate_explanation

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')

MODEL_PATH = ROOT / os.getenv('MODEL_PATH', 'models/cardiovascular_detedtion_model.pkl')
FEATURES_PATH = ROOT / 'config' / 'features.json'
POSITIVE_LABEL = os.getenv('POSITIVE_LABEL', '1')

st.set_page_config(page_title='Cardiovascular Screening Demo', page_icon='🫀', layout='centered')
st.title('🫀 Cardiovascular Screening Demo')
st.caption('Research/portfolio demonstration only. Not a medical diagnosis or a clinical decision tool.')

if not MODEL_PATH.is_file():
    st.error(f'Model file not found: {MODEL_PATH}')
    st.info('Place your existing .pkl model in models/ and update MODEL_PATH in .env if needed.')
    st.stop()

try:
    schema = json.loads(FEATURES_PATH.read_text(encoding='utf-8'))
except (OSError, ValueError) as exc:
    st.error(f'Cannot read config/features.json: {exc}')
    st.stop()

if not schema.get('verified', False):
    st.warning('config/features.json contains EXAMPLE fields, not your actual model inputs. '
               'Update them to match the features and encodings used in training, '
               'then set "verified": true.')
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error('The model could not be loaded. Make sure this is your own trusted model '
             'and your Python/scikit-learn versions match the training environment.')
    st.exception(exc)
    st.stop()


def input_widget(spec: dict):
    """Render one field and return its value without changing feature names."""
    name = spec['name']
    label = spec.get('label', name)
    kind = spec['type']
    help_text = spec.get('help')
    if kind == 'integer':
        kw = {'value': int(spec.get('default', 0)), 'step': 1, 'help': help_text, 'key': name}
        if 'min' in spec:
            kw['min_value'] = int(spec['min'])
        if 'max' in spec:
            kw['max_value'] = int(spec['max'])
        return int(st.number_input(label, **kw))
    if kind == 'float':
        kw = {'value': float(spec.get('default', 0)), 'step': float(spec.get('step', 0.1)),
              'help': help_text, 'key': name}
        if 'min' in spec:
            kw['min_value'] = float(spec['min'])
        if 'max' in spec:
            kw['max_value'] = float(spec['max'])
        return float(st.number_input(label, **kw))
    if kind == 'select':
        options = spec['options']  # e.g. [{"label":"No", "value":0}, ...]
        values = [item['value'] for item in options]
        labels = {str(item['value']): item['label'] for item in options}
        return st.selectbox(label, values,
                            format_func=lambda v: labels[str(v)],
                            help=help_text, key=name)
    raise ValueError(f'Unsupported type {kind!r} for feature {name!r}')


st.subheader('Enter the model inputs')
try:
    with st.form('screening_form'):
        feature_values = {spec['name']: input_widget(spec) for spec in schema['features']}
        submitted = st.form_submit_button('Run screening', type='primary')
except (KeyError, TypeError, ValueError) as exc:
    st.error(f'Invalid features.json configuration: {exc}')
    st.stop()

if submitted:
    st.session_state.pop('ai_explanation', None)
    try:
        st.session_state['screening_result'] = predict(model, feature_values, POSITIVE_LABEL)
    except Exception as exc:
        st.session_state.pop('screening_result', None)
        st.error('Prediction failed. Check feature names, data types, categorical encodings, '
                 'and whether preprocessing was included in the saved pipeline.')
        st.exception(exc)

result = st.session_state.get('screening_result')
if result:
    st.divider()
    st.subheader('Screening result')
    if result['flagged']:
        st.warning(f'Model predicted the positive class ({result["predicted_label"]}). '
                   'This does not confirm cardiovascular disease.')
    else:
        st.info(f'Model predicted the non-positive class ({result["predicted_label"]}). '
                'This does not rule out cardiovascular disease.')
    if result['positive_class_score'] is not None:
        st.metric('Model score for positive class', f'{result["positive_class_score"]:.1%}')
        st.caption('This is a model probability estimate, not a validated clinical risk percentage.')

    st.caption('AI text is optional. Only the binary model result is sent to the LLM, '
               'not individual measurements. The explanation is general educational content.')






    st.subheader("AI Health Assistant")

    if st.button("Generate AI Explanation"):
        try:
            with st.spinner("Generating AI explanation..."):

                explanation = generate_explanation(
                    prediction=int(result["flagged"])
                )

                # Save explanation in session state
                st.session_state["ai_explanation"] = explanation

        except Exception as e:
            st.error(f"AI explanation failed: {e}")

    # Display AI explanation
    explanation = st.session_state.get("ai_explanation")

    if explanation:
        st.subheader("AI Explanation")
        st.write(explanation)

    st.divider()
    st.info('Important: If you have chest pain, severe shortness of breath, '
            'fainting, or other emergency symptoms, seek urgent medical care. '
            'Do not rely on this application to make medical decisions.')
