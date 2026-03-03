from dotenv import load_dotenv
import os

# Load .env locally
load_dotenv()

# Fetch API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY missing! Set it in .env or Streamlit Secrets.")

os.environ["GROQ_API_KEY"] = groq_api_key
