class PromptBuilder:
    """
    Builds prompts for the RAG system.
    """

    SYSTEM_INSTRUCTION = """
You are a document question-answering assistant.

Answer the user's question using ONLY the information
provided in the retrieved context.

Rules:
1. Do not invent or assume information.
2. If the answer cannot be found in the context,
   clearly say that the information is not available
   in the provided documents.
3. Give a concise and accurate answer.
4. When possible, mention the source document.
"""

    def build(
        self,
        question: str,
        contexts: list
    ) -> str:
        """
        Build a grounded RAG prompt.
        """

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if not contexts:
            context_text = (
                "No relevant information was retrieved."
            )
        else:
            context_parts = []

            for index, result in enumerate(
                contexts,
                start=1
            ):
                payload = result.payload

                source = payload.get(
                    "file_name",
                    "Unknown source"
                )

                content = payload.get(
                    "content",
                    ""
                )

                context_parts.append(
                    f"""
SOURCE {index}: {source}

{content}
"""
                )

            context_text = "\n".join(
                context_parts
            )

        return f"""
{self.SYSTEM_INSTRUCTION}

RETRIEVED CONTEXT:
------------------
{context_text}
------------------

USER QUESTION:
{question}

ANSWER:
"""