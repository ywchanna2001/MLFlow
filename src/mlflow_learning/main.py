import mlflow
from google import genai
from dotenv import load_dotenv
from google.genai import types

load_dotenv()

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Gemini Prompts")

mlflow.gemini.autolog()

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What's the weather like in Seattle?",
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful weather assistant.",
    ),
)

print(response.text)
