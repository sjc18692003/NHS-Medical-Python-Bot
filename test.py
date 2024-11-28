import os
import streamlit as st
from bs4 import BeautifulSoup
import requests
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model="gpt-4", api_key=api_key)

# Function to scrape NHS Inform data
def scrape_nhs_data():
    base_url = "https://www.nhsinform.scot/symptoms-and-self-help/a-to-z/"
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "html.parser")
    conditions = {}
    
    for link in soup.select("a[href^='/symptoms-and-self-help/conditions/']"):
        condition_name = link.text.strip()
        condition_url = f"https://www.nhsinform.scot{link['href']}"
        condition_page = requests.get(condition_url)
        condition_soup = BeautifulSoup(condition_page.text, "html.parser")
        description = condition_soup.find("div", {"class": "editor"}).text.strip()
        conditions[condition_name] = description
    
    return conditions

# Scrape and store data
data = scrape_nhs_data()

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings()
texts = list(data.values())
keys = list(data.keys())
vectorstore = FAISS.from_texts(texts, embeddings, metadatas=[{"title": key} for key in keys])

# Define retrieval mechanism
retriever = vectorstore.as_retriever()

# Define prompt template
template = """
You are a healthcare assistant. Use the retrieved context below to provide advice:
Retrieved Context: {retrieved_context}
User Query: {input}
"""
prompt_template = PromptTemplate(template=template)

# Define RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
)

# Streamlit app
st.subheader("NHS Inform Assistant")
query = st.text_input("Ask a health-related question:")
if query:
    with st.spinner("Fetching response..."):
        result = qa_chain.run(query)
        st.write(result)
