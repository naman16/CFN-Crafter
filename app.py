import time, unstructured
import streamlit as st
from index import conversational_chain

# Custom image for the app icon and the assistant's avatar
terraform_aws_logo = 'TerraformAWS.png'

# Configure streamlit page
st.set_page_config(
    page_title="Your TerraformAWS AI Chatbot",
    page_icon=terraform_aws_logo
)

# Initialize LLM chain in session_state
if 'chain' not in st.session_state:
    st.session_state['chain']= conversational_chain()

# Initialize chat history
if 'messages' not in st.session_state:
    # Start with first message from assistant
    st.session_state['messages'] = [{"role": "assistant", 
                                  "content": "Hi human, I am TerraformAWS AI! Which AWS service can I help you with today?"}]

# Display chat messages from history on app rerun
# Custom avatar for the assistant, default avatar for user
for message in st.session_state.messages:
    if message["role"] == 'assistant':
        with st.chat_message(message["role"], avatar=terraform_aws_logo):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat logic
if query := st.chat_input("Ask me anything"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant", avatar=terraform_aws_logo):
        message_placeholder = st.empty()
        # Send user's question to our chain
        result = st.session_state['chain']({"question": query})
        response = result['answer']
        full_response = ""

        # Simulate stream of response with milliseconds delay
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})