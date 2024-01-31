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
    

# load OpenAI API key
openai_key = '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'
# load up OpenAI Chat model and embeddings
llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-4-turbo-preview")
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    

# Method to load docs / knowledge base and split into smaller chunks
def load_and_split_docs():
    loader = DirectoryLoader("data/", glob="**/*.txt", show_progress=True)
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

if __name__ == "__main__":
    docs = load_and_split_docs()
    vectorstore_faiss = creat_and_store_vector_embeddings()