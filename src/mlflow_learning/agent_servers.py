import uuid
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from mlflow.genai.agent_server impoort invoke
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

load_dotenv()  # GEMINI_API_KEY in .env

SYSTEM_PROMPT = """You provide help with math problems. Explain your reasoning step by step. If you don't know the answer, say "I don't know"."""

model = init_chat_model(
    model="gemini-2.5-flash")

async def non_streaming(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """Non-streaming agent server endpoint."""
    # Create a unique session ID for this request
    session_id = str(uuid.uuid4())

    # Create a list of messages for the chat model
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=request.input),
    ]

    # Invoke the chat model with the messages
    response = await invoke(model, messages)

    # Return the response in the expected format
    return ResponsesAgentResponse(
        session_id=session_id,
        output=response.content,
        metadata={"model": "gemini-2.5-flash"}
    )