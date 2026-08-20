# AI-Powered Document Processing and RAG System

An AI-powered document processing and Retrieval-Augmented Generation (RAG) system that allows users to upload documents, extract and process their content, store semantic embeddings, and ask natural-language questions about the uploaded documents.

The system supports PDF and image-based documents and uses Docling for document parsing, Qdrant for vector storage, and Groq for LLM-based response generation.

---

## Features

- PDF, JPG, JPEG, and PNG document support
- Document parsing using Docling
- Text extraction and preprocessing
- Intelligent text chunking
- Local embedding generation
- Vector storage using Qdrant
- Semantic document retrieval
- Document reranking
- Groq-based LLM response generation
- Source-aware answers
- FastAPI backend
- Streamlit user interface
- Local vector database storage
- Modular and extensible OOP architecture

---

## System Architecture

```text
                    User
                     |
                     v
              Streamlit UI
                     |
              HTTP Requests
                     |
                     v
               FastAPI API
                /        \
               /          \
              v            v
          /ingest         /ask
             |              |
             v              v
       Document Parser    RAG Pipeline
             |              |
             v              v
          Chunking       Retrieval
             |              |
             v              v
        Embeddings       Reranking
             |              |
             v              v
           Qdrant         Groq LLM
             |              |
             |              v
             |           Answer
             |              |
             +--------------+
                    |
                    v
                  User
asmt/
│
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── ingestion_dependencies.py
│   │   ├── routes.py
│   │   └── schemas.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedding_service.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base_llm.py
│   │   └── llm_service.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── document.py
│   │
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── base_parser.py
│   │   ├── docling_parser.py
│   │   └── parser_factory.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── chunker.py
│   │   └── text_cleaner.py
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── dependencies.py
│   │   ├── prompt.py
│   │   └── rag_pipeline.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── reranker.py
│   │   └── retriever.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_registry.py
│   │   ├── indexing_service.py
│   │   └── ingestion_service.py
│   │
│   └── vectorstore/
│       ├── __init__.py
│       └── vector_store.py
│
├── data/
│   └── extracted/
│       └── parsed_documents.json
│
├── main.py
├── streamlit_app.py
├── requirements.txt
├── .gitignore
└── README.md

# to Run ............................................
IN CMD 1 :
  >> .venv\Scripts\actiavte.bat
  >> uvicorn app.api.app:app 
IN CMD 2 :
  >> .venv\Scripts\activate.bat
  >> streamlit run streamlit_app.py
