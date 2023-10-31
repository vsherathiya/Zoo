import os
import sys
import torch
import random
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from easyllm.clients import huggingface
from src.MitraAI_API.Local_Pack.Load_model import load_model
from src.MitraAI_API.Local_Pack.data_ingestion import ingestion_main

from src.MitraAI_API.Local_Pack.prompt_template import get_prompt_template
from src import CustomException, logger
from langchain.vectorstores import Chroma
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

from src.MitraAI_API.Local_Pack.constants import (
    CHROMA_SETTINGS,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME
)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Set a random seed for reproducibility
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
MODEL_TYPE = 'mistral'

USE_HISTORY = False
SHOW_SOURCES = True
logger.info(f"Running on: {DEVICE_TYPE}")
logger.info(f"Display Source Documents set to: {SHOW_SOURCES}")

LLM = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID,
                 model_basename=MODEL_BASENAME)
EMBEDDINGS = HuggingFaceInstructEmbeddings(
    model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})


# set env for prompt builder
os.environ["HUGGINGFACE_PROMPT"] = "llama2" # vicuna, wizardlm, stablebeluga, open_assistant
os.environ["HUGGINGFACE_TOKEN"] = "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" 
# Changing configuration without using environment variables
huggingface.api_key="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"
huggingface.prompt_builder = "llama2"

MODEL = "mistralai/Mistral-7B-Instruct-v0.1"


def formate_answer(ans):
    formated_answer = huggingface.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {
                    "role": "system",
                   "content":"your task is to return only html of below text : "                   
                            +ans+
                    "use Html tags like <ul>, <li>, <br>,<p> , <b> , <h1>, <h2>, <h3> , <h4>, <h5>, <h6>, <i> for formating"
                    #"content": "your task is to find category from text like user want to make camping reservation or get camping information. Return only category name like information, reservation or other."
                    },
                    {
                    "role": "user",
                    "content": ans
                    }
                ],
                temperature=0,
                max_tokens=256
            )
    return formated_answer['choices'][0]['message']['content']


def ProcessResponse(user_prompt, prev_message_data=None):
    global DB
    global RETRIEVER
    global QA
    try:
        if not os.path.exists(PERSIST_DIRECTORY):
            print("The Vectors directory does not exist")
            ingestion_main()
            logger.info('Vectors Created')        # load the vectorstore
        # =================================================================================

        def retrieval_qa_pipline(
            device_type, use_history,
            promptTemplate_type="llama"):
        

            embeddings = HuggingFaceInstructEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": DEVICE_TYPE}
                )
        
            # load the vectorstore
            db = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings,
                client_settings=CHROMA_SETTINGS
            )
            retriever = db.as_retriever()

            # get the prompt template and memory if set by the user.
            prompt, memory = get_prompt_template(
                promptTemplate_type=promptTemplate_type, history=USE_HISTORY)

            # load the llm pipeline
            llm = load_model(DEVICE_TYPE, model_id=MODEL_ID,
                             model_basename=MODEL_BASENAME, LOGGING=logger)

            if USE_HISTORY:
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",  
                    # try other chains types as well. refine, map_reduce,
                    # map_rerank
                    retriever=retriever,
                    return_source_documents=True,  
                    verbose=False,
                    callbacks=callback_manager,
                    chain_type_kwargs={"prompt": prompt, "memory": memory},
                )
            else:
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,  
                    verbose=False,
                    callbacks=callback_manager,
                    chain_type_kwargs={
                        "prompt": prompt,
                    },
                )

            return qa

        embeddings = HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": DEVICE_TYPE}
            )
        db_for_similarity = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS
        )
        # retriever = db.as_retriever()
        qa = retrieval_qa_pipline(
            DEVICE_TYPE, USE_HISTORY, promptTemplate_type=MODEL_TYPE)
        # Interactive questions and answers

        # Get the answer from the chain
        response = qa(user_prompt)
        answer, docs = response["result"], response["source_documents"]

        # docs_and_scores = db_for_similarity.
        # similarity_search_with_relevance_scores(answer)
        docs_and_scores1 = db_for_similarity.similarity_search_with_relevance_scores(
            user_prompt)
        if docs_and_scores1[0][1] < 0.67:  # & docs_and_scores1[0][1] < 0.72:
            answer = "i don't know"
            print("\n> similarity score of query:")
            print(docs_and_scores1[0][1])
        elif 0.67 <= docs_and_scores1[0][1] <= 0.75:
            response = qa(user_prompt)
            Temp_answer, docs = response["result"], response["source_documents"]

            docs_and_scores2 = db_for_similarity.similarity_search_with_relevance_scores(
                Temp_answer)
            # & docs_and_scores2[0][1] < 0.72:
            if docs_and_scores2[0][1] < 0.75:
                answer = "i don't know"

                print("\n> similarity score of answer:")
                print(docs_and_scores2[0][1])
                print("\n> similarity score of query:")
                print(docs_and_scores1[0][1])
            else:
                answer = Temp_answer
                print("\n> similarity score of answer:")
                print(docs_and_scores2[0][1])
                print("\n> similarity score of query:")
                print(docs_and_scores1[0][1])
        else:
            answer = answer

            print("\n> similarity score of query:")
            print(docs_and_scores1[0][1])

        # Print the result
        print("\n\n> Question:")
        print(user_prompt)
        print("\n> Answer:")
        print(answer)

        # =================================================================================
        
        formatted_text = formate_answer(answer)
        if answer.lower() == "i don't know":
            formatted_text = "Unfortunately, our system couldn't find an \
                answer. We apologize. If related to Camping & Lodging and \
                    Hearst Castle, our support will contact you. \
                        Please share your email."
        return {"category": "information", "message": formatted_text , "data": {
                "place": 'null',
                "date": 'null',
                "nights": 'null'
                }}

    except Exception as e:
        raise (CustomException(e, sys))
