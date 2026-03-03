from dotenv import load_dotenv
import os

# load configuration fot environment variables
import os
from dotenv import load_dotenv
load_dotenv() ## aloading all the environment variable

# os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

import os

# Load GROQ API key from environment (Streamlit secrets are automatically loaded)
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set! Check your Streamlit secrets.")
os.environ["GROQ_API_KEY"] = groq_api_key
