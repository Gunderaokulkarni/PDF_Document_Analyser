# config.py
from dotenv import load_dotenv
import os

# Load environment variables from .env (for local dev)
load_dotenv()

# Fetch the API key safely
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY is missing! "
        "Set it in a .env file for local development, "
        "or in Streamlit Secrets for deployment."
    )

# Assign to os.environ if needed
os.environ["GROQ_API_KEY"] = groq_api_key
