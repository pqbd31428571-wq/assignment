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

            # If the paragraph fits in the current chunk
            candidate = (
                f"{current_text}\n\n{paragraph}"
                if current_text
                else paragraph
            )

            if len(candidate) <= self.chunk_size:
                current_text = candidate
                continue

            # Save the current chunk
            if current_text:
                chunks.append(
                    self._create_chunk(
                        document,
                        current_text,
                        chunk_number
                    )
                )

                chunk_number += 1

            # Create overlap from the previous chunk
            overlap_text = current_text[
                -self.chunk_overlap:
            ]

            # Start new chunk with overlap
            current_text = (
                f"{overlap_text}\n\n{paragraph}"
            ).strip()

            # If the paragraph itself is too large,
            # split it into smaller pieces.
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

        # Store final chunk
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
        """

        parts = []

        start = 0

        while start < len(text):

            end = start + self.chunk_size

            part = text[start:end].strip()

            if part:
                parts.append(part)

            if end >= len(text):
                break

            start = end - self.chunk_overlap

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