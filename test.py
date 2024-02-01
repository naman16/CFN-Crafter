from index import conversational_chain

chain = conversational_chain()
chat_history = []
query = "Write cloudformation for s3 following best practices for security"
result = chain({"question":query,"chat_history":chat_history})
print (result["answer"])
