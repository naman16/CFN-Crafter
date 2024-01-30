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
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate,MessagesPlaceholder
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

system_message_template = SystemMessagePromptTemplate.from_template(template="""
You are a cloud engineer that is an expert in Terraform for AWS and you are able to write full blown, production ready Terraform modules based on user requirements. 
If the answer is not in the context say "Sorry, I don't know, as the answer was not found in the context."
""")

human_message_template = HumanMessagePromptTemplate.from_template(template="{input}")

# Method to load docs / knowledge base and split into smaller chunks
def load_and_split_docs():
    loader = DirectoryLoader("data/", glob="output1.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 200,
    )
    docs = text_splitter.split_documents(documents)
    return docs

# Method to store chunks into a vectore store (FAISS)
def creat_and_store_vector_embeddings():
    vectorstore_faiss = FAISS.from_documents(
        docs,
        embeddings,
    )
    vectorstore_faiss.save_local("local_index")
    return vectorstore_faiss

# Condense the chat history and follow-up question into a standalone question
def conversational_retrieval_prompts_template():
    _template = """{chat_history}

Answer only with the new question.
How would you ask the question considering the previous conversation: {question}
Question:"""
    CONVO_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    return CONVO_QUESTION_PROMPT

if __name__ == "__main__":
    docs = load_and_split_docs()
    vectorstore_faiss = creat_and_store_vector_embeddings()
    wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)
    memory_chain = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)
    chat_history=[]
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore_faiss.as_retriever(),
        memory=memory_chain,
        condense_question_prompt=conversational_retrieval_prompts_template(),
        chain_type='map_reduce'
    )
    prompt_template=ChatPromptTemplate.from_messages([system_message_template, MessagesPlaceholder(variable_name="history"), human_message_template])
    qa.combine_docs_chain.llm_chain.prompt = PromptTemplate.from_template(prompt_template)
    qa.predict(input="How to fix my car?")
    