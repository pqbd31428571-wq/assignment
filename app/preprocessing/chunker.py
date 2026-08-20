from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """
    Represents a chunk of text extracted from a document.
    """

    chunk_id: str
    document_id: str
    content: str
    metadata: dict


class TextChunker:
    """
    Splits documents into overlapping chunks while
    trying to preserve paragraph boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document) -> list[DocumentChunk]:
        """
        Split a document into overlapping chunks.
        """

        text = document.content.strip()

        if not text:
            return []

        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        chunks = []
        current_text = ""
        chunk_number = 1

        for paragraph in paragraphs:

            candidate = (
                f"{current_text}\n\n{paragraph}"
                if current_text
                else paragraph
            )

            if len(candidate) <= self.chunk_size:
                current_text = candidate
                continue

            if current_text:
                chunks.append(
                    self._create_chunk(
                        document,
                        current_text,
                        chunk_number
                    )
                )

                chunk_number += 1

            overlap_text = current_text[
                -self.chunk_overlap:
            ]

            current_text = (
                f"{overlap_text}\n\n{paragraph}"
            ).strip()

            if len(current_text) > self.chunk_size:
                large_parts = self._split_large_text(
                    current_text
                )

                for part in large_parts[:-1]:
                    chunks.append(
                        self._create_chunk(
                            document,
                            part,
                            chunk_number
                        )
                    )
                    chunk_number += 1

                current_text = large_parts[-1]

        if current_text:
            chunks.append(
                self._create_chunk(
                    document,
                    current_text,
                    chunk_number
                )
            )

        return chunks

    def _split_large_text(
        self,
        text: str
    ) -> list[str]:
        """
        Split text that is larger than the chunk size.

        Splits at the nearest whitespace before the chunk_size
        boundary rather than an arbitrary character index, so a word
        (or a table cell like "TCP, UDP, SCTP") isn't cut in half by
        landing exactly on the size limit.
        """

        parts = []
        start = 0

        while start < len(text):

            end = start + self.chunk_size

            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary == -1 or boundary <= start:
                    boundary = end
            else:
                boundary = len(text)

            part = text[start:boundary].strip()

            if part:
                parts.append(part)

            if boundary >= len(text):
                break

            start = boundary - self.chunk_overlap
            if start < 0:
                start = 0

        return parts

    def _create_chunk(
        self,
        document,
        content: str,
        chunk_number: int
    ) -> DocumentChunk:

        chunk_id = (
            f"{document.document_id}_chunk_{chunk_number}"
        )

        metadata = {
            "document_id": document.document_id,
            "file_name": document.file_name,
            "file_type": document.file_type,
            "source_path": document.source_path,
            "chunk_number": chunk_number,
        }

        return DocumentChunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            content=content.strip(),
            metadata=metadata,
        )