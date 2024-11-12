import os
import streamlit as st
from streamlit_chat import message

# Importing modules from langchain_openai package
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferWindowMemory
)

# Importing specific modules from langchain package
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

from dotenv import load_dotenv

# Loading environment variables from .env file
load_dotenv()

# Setting subheader for the Streamlit app
st.subheader("Python Assistant ")

# Retrieving OpenAI API key from environment variables
api_key=os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model="gpt-4o-mini",api_key=api_key)

# Initializing session state variables if not already present
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    # Initializing ConversationBufferWindowMemory with a window size of 3
    st.session_state.buffer_memory=ConversationBufferWindowMemory(memory_key='history',
                                                                  k=3,
                                                                  return_messages=True)

# Defining template for prompts
template = """  Follow the below instructions using the provided context:
                always keep track of the users name,
                You are a python language expert, 
                you are to provide code solutions and answers to only python related questions
                Chat_history:
                {history}
                Current Conversation:
                """

# Creating template objects for system and human messages
system_msg_template = SystemMessagePromptTemplate.from_template(template=template)
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

# Creating chat prompt template from system and human message templates
prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

# Creating ConversationChain instance
conversation = ConversationChain(
     memory=st.session_state.buffer_memory,
     prompt=prompt_template,
     llm=llm,
     verbose=True)

# Container for displaying chat history
response_container = st.container()

# Container for user input textbox
query_container = st.container()

# Displaying user input textbox and handling user queries
with query_container:
    query = st.text_input('Query: ', key='input')
    if query:
        with st.spinner("typing...."):
            # Generating response for user query
            response = conversation.predict(input=f"\n{query}\n")
        # Appending user query and generated response to session state
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

# Displaying chat history and user queries along with responses
with response_container:
     if st.session_state['responses']:
          for i in range(len(st.session_state['responses'])):
               # Displaying AI responses
               message(st.session_state['responses'][i], key=str(i))
               # Displaying user queries
               if i < len(st.session_state['requests']):
                    message(st.session_state['requests'][i], is_user=True, key=str(i)+ '_user')
