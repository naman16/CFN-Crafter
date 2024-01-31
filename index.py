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
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate


'''
The below function initializes and configures a conversational retrieval chain for
answering user questions.
'''
def conversational_chain():
    # load OpenAI API key
    openai_key = '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'
    # load up OpenAI Chat model and embeddings
    llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-4-turbo-preview")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    
    # Load the local FAISS index as a retriever
    vector_store = FAISS.load_local("local_index", embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Enable conversational memory to enable chat functionality 
    memory = ConversationBufferWindowMemory(k=3,memory_key="chat_history")

    # Create the chain
    chain = ConversationalRetrievalChain.from_llm(llm=llm, 
                                                  retriever=retriever, 
                                                  memory=memory, 
                                                  get_chat_history=lambda h : h
                                                  )
    system_message_prompt_template = """
    You are an AI assistant for answering questions about the Blendle Employee Handbook.
    You are given the following extracted parts of a long document and a question. Provide a conversational answer.
    If you don't know the answer, just say 'Sorry, I don't know... 😔'.
    Don't try to make up an answer.
    If the question is not about the Blendle Employee Handbook, politely inform them that you are tuned to only answer questions about the Blendle Employee Handbook.

    {context}
    Question: {question}
    Helpful Answer:
    """
    