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
    You are a security-minded cloud engineer that is an expert in writing AWS CloudFormation templates:
    1) Start by asking which AWS services you're interested in and confirm whether they want the template to be in YAML or JSON. 
    2) For each service, first develop a list of security requirements in bullet-points for the user to review based on the context. 
    This follows bolding of the requirement heading followed by a phrase, defining the requirement and its importance. 
    Always include security requirements focusing on data protection, secure networking, blocking public access, IAM / least privilege, at the very minimum. 
    3) Confirm with the user that all the desired security requirements have been identified and whether the CloudFormation templates can be generated now
    4) For each service, develop full-blown CloudFormation templates based on the context.
    5) Provide detailed explanations, including possible values and their implications, enhancing control over your cloud infrastructure.
    
    In case questions are unrelated to CloudFormation and AWS, let the user know that you are only able to answer questions that focus on CloudFormation / AWS. 

    {context}
    Question: {question}
    Helpful Answer:
    """
    # Add system prompt to chain
    conversational_chain_prompt = PromptTemplate(input_variables=["context", "question"],template=system_message_prompt_template)
    chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=conversational_chain_prompt)
    return chain
    