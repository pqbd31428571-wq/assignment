import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentRegistry:
    """
    Keeps track of documents that have already been indexed.
    """

    def __init__(self, registry_path: str = "data/processed/document_registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        if self.registry_path.exists():
            with self.registry_path.open("r", encoding="utf-8") as file:
                self.documents = json.load(file)
            logger.info("Loaded document registry: %d entrie(s).", len(self.documents))
        else:
            self.documents = {}
            logger.info("No existing document registry found — starting fresh.")

    def _save(self):
        with self.registry_path.open("w", encoding="utf-8") as file:
            json.dump(self.documents, file, indent=4)

    def calculate_hash(self, file_path: Path) -> str:
        sha256 = hashlib.sha256()

        with file_path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                sha256.update(chunk)

        return sha256.hexdigest()

    def is_indexed(self, file_path: Path) -> bool:
        file_hash = self.calculate_hash(file_path)
        return file_hash in self.documents

    def register(self, file_path: Path, document_id: str):
        file_hash = self.calculate_hash(file_path)

        self.documents[file_hash] = {
            "document_id": document_id,
            "file_name": file_path.name,
            "source_path": str(file_path),
        }

        self._save()

        logger.info("Registered %s as indexed (document_id=%s).", file_path.name, document_id)