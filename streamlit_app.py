import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000"


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Document RAG Assistant",
    layout="wide"
)


st.title("Document RAG Assistant")

st.write(
    "Upload your documents and ask questions "
    "based on their content."
)


# --------------------------------------------------
# DOCUMENT UPLOAD
# --------------------------------------------------

st.header("1. Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDF, JPG, JPEG, or PNG files",
    type=["pdf", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)


if st.button("Process Documents"):

    if not uploaded_files:

        st.warning(
            "Please upload at least one document."
        )

    else:

        progress = st.progress(0)

        successful = 0
        failed = 0

        for index, uploaded_file in enumerate(uploaded_files):

            try:

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    f"{API_URL}/ingest",
                    files=files,
                    timeout=300
                )

                if response.status_code == 200:

                    successful += 1

                    result = response.json()

                    st.success(
                        f"✓ {uploaded_file.name} processed "
                        f"({result['chunks_indexed']} chunks)"
                    )

                else:

                    failed += 1

                    st.error(
                        f"✗ {uploaded_file.name}: "
                        f"{response.text}"
                    )

            except requests.RequestException as exc:

                failed += 1

                st.error(
                    f"✗ {uploaded_file.name}: "
                    f"Could not connect to API. {exc}"
                )

            progress.progress(
                (index + 1) / len(uploaded_files)
            )

        st.info(
            f"Completed: {successful} successful, "
            f"{failed} failed."
        )


# --------------------------------------------------
# BULK INGESTION
# --------------------------------------------------

st.subheader("Bulk Document Processing")

st.write(
    "Process all supported documents already placed "
    "in the data/raw/ directory."
)


if st.button("Ingest All"):

    with st.spinner(
        "Scanning data/raw/ and processing documents..."
    ):

        try:

            response = requests.post(
                f"{API_URL}/ingest-directory",
                timeout=3600
            )

            if response.status_code == 200:

                result = response.json()

                st.success(
                    f"Found {result['documents_found']} "
                    f"new document(s), indexed "
                    f"{result['chunks_indexed']} chunks. "
                    f"{result['message']}"
                )

            else:

                st.error(
                    f"Bulk ingestion failed: "
                    f"{response.text}"
                )

        except requests.RequestException as exc:

            st.error(
                f"Could not connect to FastAPI: {exc}"
            )


st.divider()


# --------------------------------------------------
# QUESTION ANSWERING
# --------------------------------------------------

st.header("2. Ask a Question")

question = st.text_input(
    "Enter your question",
    placeholder="Ask something about your documents..."
)


if st.button("Ask"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        try:

            response = requests.post(
                f"{API_URL}/ask",
                json={
                    "question": question
                },
                timeout=120
            )

            if response.status_code == 200:

                result = response.json()

                st.subheader("Answer")

                st.write(
                    result["answer"]
                )

                st.subheader("Sources")

                if result["sources"]:

                    for source in result["sources"]:

                        st.write(
                            f"📄 **{source['file_name']}**  \n"
                            f"Chunk: `{source['chunk_id']}`  \n"
                            f"Score: `{source['score']:.4f}`"
                        )

                else:

                    st.info(
                        "No relevant sources were found."
                    )

            else:

                st.error(
                    f"API Error: {response.text}"
                )

        except requests.RequestException as exc:

            st.error(
                f"Could not connect to FastAPI: {exc}"
            )