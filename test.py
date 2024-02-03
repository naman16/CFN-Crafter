from index import conversational_chain

chain = conversational_chain()
chat_history = []
query = "can you create cloudformation for RDS in YAML"
result = chain({"question":query,"chat_history":chat_history})
print (result["answer"])