from pathlib import Path
import json

from app.services.ingestion_service import IngestionService


RAW_DATA_DIR = Path("data/raw")
EXTRACTED_DATA_DIR = Path("data/extracted")


def main():
    """
    Main entry point of the document ingestion pipeline.
    """

    EXTRACTED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ingestion_service = IngestionService()

    documents = ingestion_service.ingest_directory(
        RAW_DATA_DIR
    )

    output_file = (
        EXTRACTED_DATA_DIR /
        "parsed_documents.json"
    )

    # Convert Document objects into dictionaries
    documents_data = [
        document.__dict__
        for document in documents
    ]

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            documents_data,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\n--------------------------------")
    print("Document ingestion completed")
    print("--------------------------------")
    print(f"Documents processed: {len(documents)}")
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    main()