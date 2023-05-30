import streamlit as st
import openai
import os
import tiktoken
from utils import system_message, ensure_fit_tokens, user_msg_container_html_template, bot_msg_container_html_template
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


secrets = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = secrets


# Load the existing persisted database from disk.
persist_path_full = "model/chromadb"
embeddings = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_path_full, embedding_function=embeddings)
vectordb_retriver = vectordb.as_retriever(search_kwargs={"k":3})

# Initiate session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Construct messages from chat history 
def construct_messages(history):
    messages = [{"role": "system", "content": system_message}]
    
    for entry in history:
        role = "user" if entry["is_user"] else "assistant"
        messages.append({"role":role, "content": entry["message"]})
   # Ensure total tokens do not exceed model's limit
    messages = ensure_fit_tokens(messages)
    return messages 
  
                        
# Function to generate response
def generate_response():
    # Append user's query to history
    st.session_state.history.append({
        "message": st.session_state.prompt,
        "is_user": True
    })
    
    
    # Construct messages from chat history
    messages = construct_messages(st.session_state.history)
    
    # Add the new_message to the list of messages before sending it to the API
    # messages.append(new_message)
    
    # Ensure total tokens do not exceed model's limit
    messages = ensure_fit_tokens(messages)
    
    # Call the Chat Completions API with the messages
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
        # stream=True
    )

    # Extract the assistant's message from the response
    assistant_message = response['choices'][0]['message']['content']
    
    # Append assistant's message to history
    st.session_state.history.append({
        "message": assistant_message,
        "is_user": False
    })
# Take user input
st.title("Glassnode Chatbot Demo")

st.text_input("Enter your prompt:",
              key="prompt",
              placeholder="e.g. '皮肤角质层是什么？'",
              on_change=generate_response
              )

# Display chat history
for message in st.session_state.history:
    if message["is_user"]:
        st.write(user_msg_container_html_template.replace("$MSG", message["message"]), unsafe_allow_html=True)
    else:
        st.write(bot_msg_container_html_template.replace("$MSG", message["message"]), unsafe_allow_html=True)
                        