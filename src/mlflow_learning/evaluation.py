import mlflow
from mlflow.genai import scorer
from mlflow.genai import scorers
from mlflow.genai.scorers import Guidelines, Correctness

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()  # GEMINI_API_KEY in .env

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Gemini Evaluation")
mlflow.gemini.autolog()

client = genai.Client()


def ask_agent(question: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful assistant. Answer questions concisely.",
        ),
    )
    return response.text


eval_dataset = [
    {
        "inputs": {"question": "What is the capital of France?"},
        "expectations": {"expected_response": "Paris"},
    },
    {
        "inputs": {"question": "Who was the first person to build an airplane?"},
        "expectations": {"expected_response": "Wright Brothers"},
    },
    {
        "inputs": {"question": "Who wrote Romeo and Juliet?"},
        "expectations": {"expected_response": "William Shakespeare"},
    }
    
]

@scorer
def is_concise(outputs: str) -> bool:
    """Evaluate if response is concise (less than 5 words)."""
    return len(outputs.split()) < 5



scorers = [Correctness(), Guidelines(name='is_english', guidelines='Answer must be provided in English.'), is_concise]
mlflow.genai.evaluate(data=eval_dataset, predict_fn=ask_agent, scorers=scorers,)