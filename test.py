from index import conversational_chain

chain = conversational_chain()
chat_history = []
query = "What is s3?"
result = chain({"question":query,"chat_history":chat_history})
print (result["answer"])