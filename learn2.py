import os
import streamlit as st
from streamlit_chat import message
import json
from bs4 import BeautifulSoup
import requests

from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
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
if not api_key:
    st.error("OpenAI API key is not set. Please configure the environment.")
    st.stop()

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME"), api_key=api_key)

# Function to scrape NHS health conditions data
@st.cache_data(show_spinner=False)
def scrape_nhs_data():
    headers = {"User-Agent": "Mozilla/5.0"}
    base_url = "https://www.nhs.uk/conditions/"
    response = requests.get(base_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    conditions = {}
    for link in soup.select("a[href^='/conditions/']"):
        condition_name = link.text.strip()
        condition_url = f"https://www.nhs.uk{link['href']}"
        condition_page = requests.get(condition_url, headers=headers)
        condition_soup = BeautifulSoup(condition_page.text, "html.parser")
        description_tag = condition_soup.find("div", class_="nhsuk-main-wrapper")
        description = description_tag.text.strip() if description_tag else "Description not available."
        conditions[condition_name] = description
    return conditions

@st.cache_data(show_spinner=False)
def load_or_cache_data():
    if os.path.exists("cached_data.json"):
        with open("cached_data.json", "r") as f:
            return json.load(f)
    else:
        data = scrape_nhs_data()
        with open("cached_data.json", "w") as f:
            json.dump(data, f)
        return data

data = load_or_cache_data()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
texts = list(data.values())
keys = list(data.keys())

if not texts or not keys:
    st.error("No data available for FAISS initialization.")
    st.stop()

@st.cache_resource(show_spinner=False)
def initialize_faiss():
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings)
    else:
        vectorstore = FAISS.from_texts(
            texts, embeddings, metadatas=[{"title": key} for key in keys]
        )
        vectorstore.save_local("faiss_index")
        return vectorstore

vectorstore = initialize_faiss()
retriever = vectorstore.as_retriever()

if "responses" not in st.session_state:
    st.session_state["responses"] = ["How can I assist you?"]
if "requests" not in st.session_state:
    st.session_state["requests"] = []
if "buffer_memory" not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(
        memory_key="history", k=3, return_messages=False
    )

# Correct prompt template definition
template = """
You are a healthcare assistant. Use the retrieved context below to provide advice:
Retrieved Context:
{retrieved_context}

Chat History:
{history}

Current Query:
{input}
"""

# Initialize the system and human prompt templates
system_msg_template = SystemMessagePromptTemplate.from_template(template=template)
human_msg_template = HumanMessagePromptTemplate.from_template("{input}")

# Define ChatPromptTemplate with memory placeholder
prompt_template = ChatPromptTemplate.from_messages(
    [system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template]
)
# Initialize ConversationChain with explicit inputs
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferWindowMemory(k=3),  # Memory stores the last 3 exchanges
    prompt=prompt_template,
    input_variables=["input", "retrieved_context"],  # Input variables as expected by the prompt
    verbose=True,
)
response_container = st.container()
query_container = st.container()

# Handle user input
with query_container:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("Typing..."):
            # Retrieve relevant context from FAISS
            retrieved_context = retriever.get_relevant_documents(query)
            
            # Combine retrieved text for the model
            retrieved_texts = "\n".join([doc.metadata["title"] for doc in retrieved_context])
            
            # Generate response using ConversationChain
            response = conversation.predict(
                input=query,
                retrieved_context=retrieved_texts,
            )
        
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