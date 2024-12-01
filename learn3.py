import os
import streamlit as st
from streamlit_chat import message
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit subheader
st.subheader("Healthcare Assistant")

# Initialize OpenAI LLM
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME"), api_key=api_key)

# Simplified prompt template
template = """
You are a helpful healthcare assistant. Use the chat history to maintain context and provide answers:
Chat History:
{history}

Current Query:
{input}
"""

# Define prompt templates
system_msg_template = SystemMessagePromptTemplate.from_template(template=template)
human_msg_template = HumanMessagePromptTemplate.from_template("{input}")

# ChatPromptTemplate with history
prompt_template = ChatPromptTemplate.from_messages(
    [system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template]
)

# Initialize ConversationChain
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferWindowMemory(k=3),  # Stores the last 3 exchanges
    prompt=prompt_template,
    verbose=True,
)

# Initialize session state
if "responses" not in st.session_state:
    st.session_state["responses"] = ["How can I assist you?"]
if "requests" not in st.session_state:
    st.session_state["requests"] = []

# User input container
query_container = st.container()

# Chat response container
response_container = st.container()

# Handle user input
with query_container:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("Typing..."):
            # Generate response
            response = conversation.predict(input=query)
        
        # Update session state
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

# Display chat history
with response_container:
    if st.session_state["responses"]:
        for i in range(len(st.session_state["responses"])):
            # Display bot response
            message(st.session_state["responses"][i], key=str(i))
            # Display user query
            if i < len(st.session_state["requests"]):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + "_user")
