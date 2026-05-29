import google.genai as genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model=os.getenv("GEMINI_MODEL"),
    contents="Say hello in one sentence as a media AI assistant."
)

print("Gemini connection successful!")
print("Response:", response.text)