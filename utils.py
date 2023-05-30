import streamlit as st
import os
import tiktoken
from retrying import retry


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
