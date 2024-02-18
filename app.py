import os
import streamlit as st
from streamlit_chat import message

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferWindowMemory
)

from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

from dotenv import load_dotenv

load_dotenv()

st.subheader("Python Assistant ")

api_key=os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model="gpt-3.5-turbo",api_key=api_key)

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(memory_key='history',
                                                                          k=3,
                                                                          return_messages=True)
            
#st.write(st.session_state)

template = """  Answer the question as truthfully as possible using the provided context, 
                always keep track of the users name,
                You are a python language expert,
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