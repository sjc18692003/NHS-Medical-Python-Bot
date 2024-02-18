import os
import streamlit as st
from streamlit_chat import message

from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferWindowMemory,
    CombinedMemory, 
    ConversationSummaryMemory, 
    ConversationBufferMemory
)

from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)

from langchain_community.vectorstores.chroma import Chroma
from PyPDF2 import PdfReader

from getpass import getpass
from app import *

api_key=os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-3.5-turbo")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(memory_key='history',
                                                                          k=3,
                                                                          return_messages=True)
            
st.write(st.session_state)

template = """  Make sure you remember the users name and always acknowledge it,
                You are a python language expert,
                don't answer questions outside python programming langiage, 
                always introduce yourself as such, 
                you are to provide code solutions to all python related questions
                Chat_history:
                {history}
                Current Conversation:
                """

system_msg_template = SystemMessagePromptTemplate.from_template(template=template)

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(
     memory=st.session_state.buffer_memory,
     prompt=prompt_template,
     llm=llm,
     verbose=True)

# container for chat history
response_container = st.container()

# container for text box
query_container = st.container()

with query_container:
    query = st.text_input('Query: ', key='input')
    if query:
        with st.spinner("typing...."):
            response = conversation.predict(input=f"\n{query}\n")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with response_container:
     if st.session_state['responses']:
          for i in range(len(st.session_state['responses'])):
               message(st.session_state['responses'][i], key=str(i))
               if i < len(st.session_state['requests']):
                    message(st.session_state['requests'][i], is_user=True, key=str(i)+ '_user')