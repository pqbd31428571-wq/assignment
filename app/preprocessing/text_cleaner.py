import re


class TextCleaner:
    """
    Cleans extracted document text before chunking.
    """

    def clean(self, text: str) -> str:
        """
        Clean and normalize extracted text.

        Args:
            text: Raw extracted text.

        Returns:
            Cleaned text.
        """

        if not text:
            return ""

        # Normalize line endings
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Remove excessive spaces and tabs
        text = re.sub(r"[ \t]+", " ", text)

        # Remove excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove spaces at the beginning/end of lines
        text = "\n".join(
            line.strip()
            for line in text.splitlines()
        )

        # Remove leading/trailing whitespace
        text = text.strip()

        return text