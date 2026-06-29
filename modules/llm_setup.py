"""
modules/llm_setup.py
─────────────────────
Uses Google Gemini API - works on any network, no DNS issues.
Get your FREE key at: https://aistudio.google.com/apikey
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def get_llm(temperature: float = 0.6, max_new_tokens: int = 512):
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_google_api_key_here":
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set.\n"
            "1. Go to https://aistudio.google.com/apikey\n"
            "2. Create a free API key\n"
            "3. Paste it into your .env file as GOOGLE_API_KEY=..."
        )

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
        max_output_tokens=max_new_tokens,
    )
