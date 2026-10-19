import uuid
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from mlflow.genai.agent_server import invoke
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

load_dotenv()  # GEMINI_API_KEY in .env

SYSTEM_PROMPT = """You provide help with math problems. Explain your reasoning step by step. If you don't know the answer, say "I don't know"."""

model = init_chat_model(
    model="google_genai:gemini-2.5-flash")

@invoke()
async def non_streaming(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    msgs = [SystemMessage(content=SYSTEM_PROMPT)] + [i.model_dump() for i in request.input]
    result = await model.ainvoke(msgs)


    return ResponsesAgentResponse(
        output=[
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": result.content}],
            }
        ]
    )