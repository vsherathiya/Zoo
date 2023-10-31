import os
import sys
import torch
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from src.MitraAI_API.Local_Pack.Load_model import load_model
from src.MitraAI_API.Local_Pack.data_ingestion import ingestion_main
from src.MitraAI_API.Local_Pack.prompt_template import get_prompt_template
from src import CustomException, logger
from src.MitraAI_API.Local_Pack.constants import (
    CHROMA_SETTINGS,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME
)

import random
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)

if torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
elif torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
else:
    DEVICE_TYPE = "cpu"

SHOW_SOURCES = True
logger.info(f"Running on: {DEVICE_TYPE}")
logger.info(f"Display Source Documents set to: {SHOW_SOURCES}")

LLM = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID,
                 model_basename=MODEL_BASENAME)
EMBEDDINGS = HuggingFaceInstructEmbeddings(
    model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})


def ProcessResponse(user_prompt, prev_message_data=None):
    global DB
    global RETRIEVER
    global QA
    try:
        if not os.path.exists(PERSIST_DIRECTORY):
            print("The Vectors directory does not exist")
            ingestion_main()
            logger.info('Vectors Created')        # load the vectorstore

        DB = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=EMBEDDINGS,
            client_settings=CHROMA_SETTINGS,
        )
        RETRIEVER = DB.as_retriever()

        prompt, memory = get_prompt_template(
            promptTemplate_type="llama", history=False)

        QA = RetrievalQA.from_chain_type(
            llm=LLM,
            chain_type="stuff",
            retriever=RETRIEVER,
            return_source_documents=SHOW_SOURCES,

            chain_type_kwargs={
                "prompt": prompt,
            },
        )

        response = QA(user_prompt, None)
        # answer = response["result"]

        formatted_text = response['result']
        if response['result'].lower() == "i don't know":
            formatted_text = "Unfortunately, our system couldn't find an \
                answer. We apologize. If related to Camping & Lodging and \
                Hearst Castle, our support will contact you. Please share \
                your email."
        return {"categoty": "information", "message": formatted_text, "data": {
                "place": 'null',
                "date": 'null',
                "nights": 'null'
                }}

    except Exception as e:
        raise (CustomException(e, sys))
