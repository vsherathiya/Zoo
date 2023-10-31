import os
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from src import logger
from src.MitraAI_API.Local_Pack.constants import (
    CHROMA_SETTINGS,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
    DEVICE_TYPE
)

from langchain.document_loaders import (
    CSVLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
)


# Define document types
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}


def load_and_process_documents(source_dir, embedding_model_name, device_type):
    # Step 1: Load documents from the source directory
    logger.info(f"Step 1: Loading documents from {source_dir}")
    documents = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            file_extension = os.path.splitext(file_name)[1]
            if file_extension in DOCUMENT_MAP:
                file_path = os.path.join(root, file_name)
                loader_class = DOCUMENT_MAP[file_extension]
                try:
                    loader = loader_class(file_path)
                    doc = loader.load()[0]
                    documents.append(doc)
                    logger.info(f"{file_path} loaded.")
                except Exception as ex:
                    logger.error(f"{file_path} loading error: {ex}")

    # Step 2: Split text documents into chunks
    logger.info("Step 2: Splitting text documents into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                                   chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Step 3: Create embeddings
    logger.info("Step 3: Creating embeddings")
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device_type},
    )

    # Step 4: Generate and store vectors
    logger.info("Step 4: Generating and storing vectors")
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )

    logger.info("All steps completed.")


def ingestion_main(device_type=DEVICE_TYPE):
    logger.info("Starting the ingestion process")
    load_and_process_documents(SOURCE_DIRECTORY, EMBEDDING_MODEL_NAME, device_type)
