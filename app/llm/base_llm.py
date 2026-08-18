from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Abstract interface for Large Language Models.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: Prompt sent to the model.

        Returns:
            Generated response.
        """
        raise NotImplementedError