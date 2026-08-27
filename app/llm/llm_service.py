import logging
import os

from dotenv import load_dotenv
from groq import Groq

from .base_llm import BaseLLM

logger = logging.getLogger(__name__)


class GroqLLM(BaseLLM):
    """
    LLM implementation using the Groq API.
    """

    MODEL_NAME = "openai/gpt-oss-20b"

    def __init__(self):
        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            logger.error("GROQ_API_KEY is not configured.")
            raise ValueError("GROQ_API_KEY is not configured.")

        self.client = Groq(api_key=api_key)
        logger.info("GroqLLM initialized with model '%s'", self.MODEL_NAME)

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        try:
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
        except Exception:
            logger.exception("Groq API call failed.")
            raise

        answer = response.choices[0].message.content
        logger.info("Groq generated a %d-character response.", len(answer or ""))

        return answer