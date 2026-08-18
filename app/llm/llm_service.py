import os

from dotenv import load_dotenv
from groq import Groq

from .base_llm import BaseLLM


class GroqLLM(BaseLLM):
    """
    LLM implementation using the Groq API.
    """

    MODEL_NAME = "openai/gpt-oss-20b"

    def __init__(self):
        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured."
            )

        self.client = Groq(
            api_key=api_key
        )

    def generate(self, prompt: str) -> str:
        """
        Generate a response using the Groq API.
        """

        if not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        response = self.client.chat.completions.create(
            model=self.MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=1024,
        )

        return response.choices[0].message.content