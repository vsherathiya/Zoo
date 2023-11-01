import os

from chromadb.config import Settings
from langchain.document_loaders import (CSVLoader,
                                        TextLoader,
                                        UnstructuredExcelLoader,
                                        Docx2txtLoader,
                                        UnstructuredFileLoader,
                                        UnstructuredMarkdownLoader)
import torch
import random
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)


ROOT_DIRECTORY = os.getcwd()

# Define the folder for storing database
SOURCE_DIRECTORY = os.path.join(ROOT_DIRECTORY, "content")

PERSIST_DIRECTORY = os.path.join(ROOT_DIRECTORY, "artifact", "DB")

MODELS_PATH = os.path.join(ROOT_DIRECTORY, "artifact", "models")
DEVICE_TYPE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or -1

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

# Context Window and Max New Tokens
CONTEXT_WINDOW_SIZE = 4096
MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  # int(CONTEXT_WINDOW_SIZE/4)


N_GPU_LAYERS = 100  # Llama-2-70B has 83 layers
N_BATCH = 512

DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".py": TextLoader,
    ".pdf": UnstructuredFileLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}

#=======================================Default Instructor Model =============================
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"  # Uses 1.5 GB of VRAM (High Accuracy with lower VRAM usage)

#### SELECT AN OPEN SOURCE LLM (LARGE LANGUAGE MODEL)
# Select the Model ID and model_basename
# load the LLM for generating Natural Language responses

#### GPU VRAM Memory required for LLM Models (ONLY) by Billion Parameter value (B Model)
#### Does not include VRAM used by Embedding Models - which use an additional 2GB-7GB of VRAM depending on the model.
####
#### (B Model)   (float32)    (float16)    (GPTQ 8bit)         (GPTQ 4bit)
####    7b         28 GB        14 GB       7 GB - 9 GB        3.5 GB - 5 GB
####    13b        52 GB        26 GB       13 GB - 15 GB      6.5 GB - 8 GB
####    32b        130 GB       65 GB       32.5 GB - 35 GB    16.25 GB - 19 GB
####    65b        260.8 GB     130.4 GB    65.2 GB - 67 GB    32.6 GB -  - 35 GB

# MODEL_ID = "TheBloke/Llama-2-7B-Chat-GGML"
# MODEL_BASENAME = "llama-2-7b-chat.ggmlv3.q4_0.bin"

####
#### (FOR GGUF MODELS)
####

# MODEL_ID = "TheBloke/Llama-2-13b-Chat-GGUF"
# MODEL_BASENAME = "llama-2-13b-chat.Q4_K_M.gguf"

# MODEL_ID = "TheBloke/Llama-2-7b-Chat-GGUF"
# MODEL_BASENAME = "llama-2-7b-chat.Q4_K_M.gguf"

MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
MODEL_BASENAME = "mistral-7b-instruct-v0.1.Q8_0.gguf"

