"""
check_llm.py

Confirms your Groq API key actually works before wiring an LLM call into
explanation_service.py - same idea as check_gpu.py proving CUDA before
the real training run.

    python check_llm.py
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # reads GROQ_API_KEY from backend/.env

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("GROQ_API_KEY not found. Create backend/.env with a line like:")
    print("  GROQ_API_KEY=gsk_your_key_here")
    raise SystemExit(1)

client = Groq(api_key=api_key)

# # First, just list what models this key actually has access to - avoids
# # guessing a model name that's since been renamed or retired.
# print("Available models on your account:\n")
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a media-literacy assistant that explains framing "
                "bias in one or two plain sentences, without declaring "
                "anything definitively true or false."
            ),
        },
        {
            "role": "user",
            "content": (
                "Explain, in one sentence, what makes this text potentially "
                "biased: 'The minister slammed the so-called radical "
                "opposition in a shocking outburst.'"
            ),
        },
    ],
)

print("Groq responded:\n")
print(response.choices[0].message.content)