import mlflow
from dotenv import load_dotenv
from mlflow import client


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

load_dotenv()

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Gemini Prompts")
mlflow.langchain.autolog()

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    return f"The current weather in {city} is sunny with a temperature of 25°C."

chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools([get_weather])
response = chat.invoke(
    [
        SystemMessage(content="You are a helpful weather assistant."),
        HumanMessage(content="What's the weather like in Seattle?"),
    ]
)


print(response)
