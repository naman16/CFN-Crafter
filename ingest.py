# importing necessary packages
from langchain_openai import OpenAI, ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate
import lxml
from bs4 import BeautifulSoup
import requests
import xml.etree.ElementTree as ET
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# Setup Chrome Driver, may need to change based on system
service = Service("/usr/bin/chromedriver")
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
driver = webdriver.Chrome(service=service, options=options)


# load OpenAI API key
openai_key = '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'
# load up OpenAI Chat model and embeddings
llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-4-turbo-preview")
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
sitemap_urls = [
    "https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/sitemap.xml",
    "https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/sitemap.xml"
]

def extract_urls_from_sitemap(sitemap):
    response = requests.get(sitemap)
    print (response)
    if response.status_code != 200:
        print(f"Failed to fetch sitemap: {response.status_code}")
        return []

    sitemap_content = response.content
    root = ET.fromstring(sitemap_content)

    # Extract the URLs from the sitemap
    urls = [
        elem.text
        for elem in root.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
    ]

    return urls

# Method to load docs / knowledge base and split into smaller chunks
def load_and_split_docs(urls):
    loader = SeleniumURLLoader(urls=urls)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)
    return docs

# Method to store chunks into a vectore store (FAISS)
def creat_and_store_vector_embeddings(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        embeddings,
    )
    vectorstore_faiss.save_local("local_index")
    return vectorstore_faiss

if __name__ == "__main__":
    full_urls_list =[]
    for sitemap in sitemap_urls:
        full_urls_list.extend(extract_urls_from_sitemap(sitemap))
    docs = load_and_split_docs(full_urls_list)
    vectorstore_faiss = creat_and_store_vector_embeddings(docs)