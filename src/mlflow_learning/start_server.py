import agent
import mlflow
from mlflow.genai.agent_server import AgentServer, setup_mlflow_git_based_version_tracking

mlflow.set_tracking_uri("http://localhost:5000")

agent_server = AgentServer('ResponsesAgent')
app = agent_server.app

setup_mlflow_git_based_version_tracking()
agent_server.run(app_import_string="start_server:app")