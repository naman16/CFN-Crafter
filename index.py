# importing necessary packages
from langchain_openai import OpenAI, ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from streamlit_chat import message
from langchain.schema import AIMessage, HumanMessage, format_document
import streamlit as st
import numpy as np
import random
import time

# provide OpenAI API key
openai_key = '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'

# Setting up OpenAI LLM model
llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-4-turbo-preview")

embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

# Loading knowledge base
loader = DirectoryLoader("data/", glob="output1.txt")


documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 200,
)

docs = text_splitter.split_documents(documents)

vectorstore_faiss = FAISS.from_documents(
    docs,
    embeddings,
)

vectorstore_faiss.save_local("local index")








retriever = vectorstore_faiss.as_retriever()

wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)
print(wrapper_store_faiss.query("Terraform for AWS ACM", llm=llm))

memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("You are a cloud engineer that is an expert in writing Terraform code for AWS based on the information included in the data / context provided by me")
memory.chat_memory.add_ai_message("I am career coach and give career advice")
conversation = ConversationChain(
     llm=llm, verbose=True, memory=memory
)