import mlflow

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()  # GEMINI_API_KEY in .env

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Gemini Evaluation")
mlflow.gemini.autolog()

client = genai.Client()

@mlflow.trace
def run_prompt(num_bullet_points, summary_content):
    prompt = mlflow.genai.load_prompt('prompts:/MyPrompt/2')
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt.format(num_bullet_points=num_bullet_points, summary_content=summary_content),
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful assistant. Answer questions concisely.",
        ),
    )
    return response.text

text = """In deep learning, the transformer is a family of artificial neural network architectures based on the multi-head attention mechanism, in which input data such as text, images, or audio is converted to a sequence of numerical representations called tokens, and each token is converted into a vector through an embedding layer.[1] At each layer, each token is then contextualized within the scope of the context window with other (unmasked) tokens via a parallel multi-head attention mechanism, allowing the signal for key tokens to be amplified and less important tokens to be diminished. Because self-attention alone is permutation-invariant, transformers inject positional information, typically through positional encodings or learned positional embeddings, so token order can affect the output.[1]

Because transformers do not process tokens one at a time, their computations can be parallelized across sequence positions during training more readily than those of recurrent neural networks like long short-term memory (LSTM).[2] Later variations have been widely adopted for training large language models (LLMs) on large (language) datasets.[citation needed]"""

print(run_prompt(num_bullet_points=3, summary_content=text))