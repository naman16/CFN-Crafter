def create_prompt_template():
    _template = """{chat_history}

Answer only with the new question.
How would you ask the question considering the previous conversation: {question}
Question:"""
    CONVO_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    return CONVO_QUESTION_PROMPT

memory_chain = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)
chat_history=[]

qa = ConversationalRetrievalChain.from_llm(
    llm=llm, 
    retriever=vectorstore_faiss.as_retriever(search_type='similarity', search_kwargs={"k": 8}),
    memory=memory_chain,
    #verbose=True,
    condense_question_prompt=create_prompt_template(), 
    chain_type='stuff', # 'refine',
    #max_tokens_limit=100
)