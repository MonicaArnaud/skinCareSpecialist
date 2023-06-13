import streamlit as st
import openai
import os
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import tiktoken
from retrying import retry
import asyncio

from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage




secrets = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = secrets

system_message = """
    You are skin care specialist GPT, a highly sophisticated language model trained to provide medical and care advice from the multiple articles and documents.  
    Your responses should be focused, practical, and direct, mirroring the communication styles of a skin care specialist. Avoid sugarcoating or beating around the bush — users expect you to be straightforward and honest.
    You have access to documents stored in a vector database. When a user provides a query, you will be provided with snippets of documents that may be relevant to the query. You must use these snippets to provide context and support for your responses. Rely heavily on the content of the document to ensure accuracy and authenticity in your answers.
    Be aware that the chunks of text provided may not always be relevant to the query. Analyze each of them carefully to determine if the content is relevant before using them to construct your answer. Do not make things up or provide information that is not supported by the documents.
    Always maintain the signature no-bullshit approach of the knowledge base, if you don't konw the answer, simple reply "I don't know".
    In your answers, speak confidently as if you were simply speaking from your own knowledge.
"""

# Html format chat bot
bot_msg_container_html_template = '''
<div style='background-color: #FFFFFF; padding: 10px; border-radius: 5px; margin-bottom: 10px; display: flex'>
    <div style="width: 20%; display: flex; justify-content: center">
        <img src="https://cdn.freebiesupply.com/logos/large/2x/davidoff-goodlife-woman-logo-svg-vector.svg" style="max-height: 50px; max-width: 50px; border-radius: 50%;">
    </div>
    <div style="width: 80%;">
        $MSG
    </div>
</div>
'''

user_msg_container_html_template = '''
<div style='background-color: #FFFFFF; padding: 10px; border-radius: 5px; margin-bottom: 10px; display: flex'>
    <div style="width: 78%">
        $MSG
    </div>
    <div style="width: 20%; margin-left: auto; display: flex; justify-content: center;">
        <img src="https://cdn.freebiesupply.com/images/large/2x/star-shape.jpg" style="max-width: 50px; max-height: 50px; float: right; border-radius: 50%;">
    </div>    
</div>
'''
def render_chat(**kwargs):
    """
    Handles is_user 
    """
    if kwargs["is_user"]:
        st.write(
            user_msg_container_html_template.replace("$MSG", kwargs["message"]),
            unsafe_allow_html=True)
    else:
        st.write(
            bot_msg_container_html_template.replace("$MSG", kwargs["message"]),
            unsafe_allow_html=True)

    if "figs" in kwargs:
        for f in kwargs["figs"]:
            st.plotly_chart(f, use_container_width=True)

# Define functions to limit the token consumption
def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
  """Returns the number of tokens used by a list of messages."""
  try:
      encoding = tiktoken.encoding_for_model(model)
  except KeyError:
      encoding = tiktoken.get_encoding("cl100k_base")
  if model == "gpt-3.5-turbo": 
      num_tokens = 0
      for message in messages:
          num_tokens += 4  
          for key, value in message.items():
              num_tokens += len(encoding.encode(value))
              if key == "name":  # if there's a name, the role is omitted
                  num_tokens += -1  # role is always required and always 1 token
      num_tokens += 2  # every reply is primed with <im_start>assistant
      return num_tokens
  else:
      raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

def ensure_fit_tokens(messages):
    """
    Ensure that total tokens in messages is less than MAX_TOKENS.
    If not, remove oldest messages until it fits.
    """
    total_tokens = num_tokens_from_messages(messages)
    while total_tokens > 4096:
        removed_message = messages.pop(0)
        total_tokens = num_tokens_from_messages(messages)
    return messages


# Load the existing persisted database from disk.
# persist_path_full = "model/chromadb"
# embeddings = OpenAIEmbeddings()
# vectordb = Chroma(persist_directory=persist_path_full, embedding_function=embeddings)
# vectordb_retriver = vectordb.as_retriever(search_kwargs={"k":3})

# # Initiate session state for chat history
# if "history" not in st.session_state:
#     st.session_state.history = []

# # Construct messages from chat history 
# def construct_messages(history):
#     messages = [{"role": "system", "content": system_message}]
    
#     for entry in history:
#         role = "user" if entry["is_user"] else "assistant"
#         messages.append({"role":role, "content": entry["message"]})
#    # Ensure total tokens do not exceed model's limit
#     messages = ensure_fit_tokens(messages)
#     return messages 
  
                        
# Function to generate response
# def generate_response():
#     # Append user's query to history
#     st.session_state.history.append({
#         "message": st.session_state.prompt,
#         "is_user": True
#     })
    
    
#     # Construct messages from chat history
#     messages = construct_messages(st.session_state.history)
   
    
#     # Ensure total tokens do not exceed model's limit
#     messages = ensure_fit_tokens(messages)
    
# #     # Call the Chat Completions API with the messages
    
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         #messages=messages
#         messages = messages
#    )
    
    
#     # Extract the assistant's message from the response
#     assistant_message  = response['choices'][0]['message']['content']  # Old version 
#     # assistant_message_text = response['choices'][0]['message']['content']  # New version 
     
#     #llm = OpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0) # new version
#     # assistant_message = llm(assistant_message_text)
#     # new version
    
#     # Append assistant's message to history
#     st.session_state.history.append({
#         "message": assistant_message,  #old version
#         #"message": response_text,
#         "is_user": False
#     })
### -----------------------------------------------------### 

# Load the existing persisted database from disk.
persist_path_full = "model/chromadb"
embeddings = OpenAIEmbeddings()
with st.spinner("Loading vector database..."):
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
#async def generate_response(): # new version adding the async
async def generate_response():
    # Append user's query to history
    st.session_state.history.append({
        "message": st.session_state.prompt,
        "is_user": True
    })
    
    
    # Construct messages from chat history
    messages = construct_messages(st.session_state.history)
    
    
    # Ensure total tokens do not exceed model's limit
    messages = ensure_fit_tokens(messages)
    
# # Call the Chat Completions API with the messages
    
    response_text = ""
    async for chunk in await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500,
        stream=True,
        temperature=0.5,
        stop = NLP_MODEl_REPLY_MAX_TOKENS,
        stream = True,
        timout = TIMEOUT,
    ):
        content = chunk["choices"][0].get("delta", {}).get("content", None)
        if content is not None:
            response_text += content
            with reply_box:
                get_chat_message({
                    "role": "assistant",
                    "content":content
                }, streaming = True)

            # Continuously render the reply as it comes in
            st.text(response_text)
            
    
    # Append assistant's message to history
    st.session_state.history.append({
        #"message": assistant_message,  #old version
        "message": response_text,
        "is_user": False
    })
    
def run_async_task():
    asyncio.run(generate_response())
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(generate_response())






### ---------------------------------------------------### 
st.title("皮肤护理Chatbot Demo")

# Display chat history
for message in st.session_state.history:
    if message["is_user"]:
        st.write(user_msg_container_html_template.replace("$MSG", message["message"]), unsafe_allow_html=True)
    else:
        st.write(bot_msg_container_html_template.replace("$MSG", message["message"]), unsafe_allow_html=True)
                            
    
# Take user input

st.text_input(" 请输入您的问题 ",                             
               key="prompt",
               placeholder="e.g. '皮肤角质层是什么？'",
               # on_change= generate_response    old version      
              # on_change = handle_prompt_change # new version
              )

if st.button('Generate response'):
    run_async_task()

