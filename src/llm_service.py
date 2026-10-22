
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

if not os.getenv("GROQ_API_KEY"):
    try:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except (FileNotFoundError, KeyError):
        pass

MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "llama-3.1-8b-instant"
)


def generate_explanation(prediction, probability=None):
    """
    Generate a cardiovascular screening explanation
    using Groq's Llama 3.1 8B model.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        import streamlit as st
        api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Check your .env file."
        )

    client = Groq(api_key=api_key)

    # Convert ML prediction to readable label
    prediction_label = (
        "Positive screening classification"
        if int(prediction) == 1
        else "Negative screening classification"
    )

    probability_text = (
        f"Positive-class model score: {probability:.2%}"
        if probability is not None
        else "Positive-class model score unavailable"
    )

    # Prepare the prompt
    prompt = f"""
    A cardiovascular disease machine learning
    screening model returned the following:

    Prediction: {prediction_label}
    {probability_text}

    Please provide:
    1. An explanation of the screening result.
    2. General cardiovascular health information.
    3. General healthy lifestyle suggestions.
    4. A reminder to seek professional medical advice.

    Important:
    - This is not a medical diagnosis.
    - Do not claim the person has or does not
      have cardiovascular disease.
    - Do not interpret the score as a clinically
      validated disease probability.
    - Do not recommend medication or treatment.
    - Do not invent patient measurements or symptoms.
    - Use clear, understandable English.
    - Keep the response concise.
    """

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI assistant that explains "
                    "machine learning cardiovascular screening "
                    "results for educational purposes. "
                    "You must not provide diagnoses."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=500
    )

    return completion.choices[0].message.content
