# Passthrough to Google's Gemini API. New Google features are available immediately.
# Note: endpoint name is part of the URL path.

from google import genai

# Configure with custom endpoint
client = genai.Client(
    api_key='dummy',
    http_options={
        'base_url': "http://127.0.0.1:5000/gateway/gemini",
    }
)
response = client.models.generate_content(
    model="myendpoint",
    contents={'text': 'How are you?'},
)
client.close()
print(response.candidates[0].content.parts[0].text)