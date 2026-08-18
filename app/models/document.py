from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Document:
    """
    Represents a parsed document in the RAG system.
    """

    document_id: str
    file_name: str
    file_type: str
    source_path: str
    content: str

    metadata: dict = field(default_factory=dict)