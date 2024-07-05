import streamlit as st
import os
import tempfile
import uuid
import pdfplumber  # Import pdfplumber


class DocumentProcessor:
    def __init__(self):
        self.pages = []  # List to keep track of pages from all documents

    def ingest_documents(self):
        uploaded_files = st.file_uploader("Upload PDF files", type='pdf', accept_multiple_files=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                unique_id = uuid.uuid4().hex
                original_name, file_extension = os.path.splitext(uploaded_file.name)
                temp_file_name = f"{original_name}_{unique_id}{file_extension}"
                temp_file_path = os.path.join(tempfile.gettempdir(), temp_file_name)

                with open(temp_file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                # Use pdfplumber to extract text
                with pdfplumber.open(temp_file_path) as pdf:
                    pages_text = [page.extract_text() for page in pdf.pages]
                self.pages.extend(pages_text)  # Append extracted text to pages list

                os.unlink(temp_file_path)

            st.write(f"Total pages processed: {len(self.pages)}")


if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.ingest_documents()