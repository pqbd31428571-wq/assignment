import hashlib
import json
from pathlib import Path


class DocumentRegistry:
    """
    Keeps track of documents that have already been indexed.
    """

    def __init__(
        self,
        registry_path: str = "data/processed/document_registry.json"
    ):
        self.registry_path = Path(registry_path)

        self.registry_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self._load()

    def _load(self):
        """
        Load the registry from disk.
        """

        if self.registry_path.exists():

            with self.registry_path.open(
                "r",
                encoding="utf-8"
            ) as file:
                self.documents = json.load(file)

        else:
            self.documents = {}

    def _save(self):
        """
        Save the registry to disk.
        """

        with self.registry_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.documents,
                file,
                indent=4
            )

    def calculate_hash(
        self,
        file_path: Path
    ) -> str:
        """
        Calculate SHA-256 hash of a file.

        The hash changes when the file content changes.
        """

        sha256 = hashlib.sha256()

        with file_path.open("rb") as file:

            while chunk := file.read(1024 * 1024):
                sha256.update(chunk)

        return sha256.hexdigest()

    def is_indexed(
        self,
        file_path: Path
    ) -> bool:
        """
        Check whether a file with the same content
        has already been indexed.
        """

        file_hash = self.calculate_hash(file_path)

        return file_hash in self.documents

    def register(
        self,
        file_path: Path,
        document_id: str
    ):
        """
        Register a successfully indexed document.
        """

        file_hash = self.calculate_hash(file_path)

        self.documents[file_hash] = {
            "document_id": document_id,
            "file_name": file_path.name,
            "source_path": str(file_path),
        }

        self._save()