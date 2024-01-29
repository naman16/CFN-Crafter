# importing necessary packages

from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import streamlit as st 
import numpy as np

# provide OpenAI API key
openai_key = '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'

# Setting up OpenAI LLM model
llm = OpenAI(openai_api_key=openai_key, model_name="gpt-4-turbo-preview")

embeddings = OpenAIEmbeddings(openai_api_key=openai_key)


loader = DirectoryLoader("data/", glob="**/*.txt")

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 100,
)
docs = text_splitter.split_documents(documents)
print (docs)

vectorstore_faiss = FAISS.from_documents(
    docs,
    embeddings,
)

wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)
print (wrapper_store_faiss)