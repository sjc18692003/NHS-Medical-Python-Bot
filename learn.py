import os
import json
import time
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
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME"), api_key=api_key)

# Cache data using Streamlit's caching mechanism
@st.cache_data(show_spinner=False)
def scrape_nhs_data():
    """Scrapes NHS Health A-Z data and caches the results."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    base_url = "https://www.nhs.uk/conditions/"
    response = requests.get(base_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    conditions = {}

    # Loop through all conditions listed on the page
    for link in soup.select("a[href^='/conditions/']"):
        condition_name = link.text.strip()
        condition_url = f"https://www.nhs.uk{link['href']}"
        condition_page = requests.get(condition_url, headers=headers)
        condition_soup = BeautifulSoup(condition_page.text, "html.parser")

        # Extract the description or main content
        description_tag = condition_soup.find("div", {"class": "nhsuk-main-wrapper"})
        description = description_tag.text.strip() if description_tag else "Description not available."
        conditions[condition_name] = description

    return conditions

# Load or cache the scraped data
@st.cache_data(show_spinner=False)
def load_or_cache_data():
    """Loads or scrapes NHS data and caches it."""
    if os.path.exists("cached_data.json"):
        with open("cached_data.json", "r") as f:
            return json.load(f)
    else:
        data = scrape_nhs_data()
        with open("cached_data.json", "w") as f:
            json.dump(data, f)
        return data

data = load_or_cache_data()

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Prepare texts and metadata
texts = list(data.values())
keys = list(data.keys())

# Cache embeddings and FAISS initialization
@st.cache_resource(show_spinner=False)
def initialize_faiss():
    """Initializes FAISS with cached embeddings."""
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings)
    else:
        vectorstore = FAISS.from_texts(
            texts,
            embeddings,
            metadatas=[{"title": key} for key in keys],
        )
        vectorstore.save_local("faiss_index")
        return vectorstore

vectorstore = initialize_faiss()

# Define retrieval mechanism
retriever = vectorstore.as_retriever()

# Define prompt template with proper input variables
template = """
You are a healthcare assistant. Use the retrieved context below to provide advice:
Retrieved Context: {retrieved_context}
User Query: {query}
"""
prompt_template = PromptTemplate(
    template=template,
    input_variables=["retrieved_context", "query"]
)

# Define the retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={
        "prompt": prompt_template,  # Correct prompt template
        "document_variable_name": "retrieved_context"  # Ensures alignment with the variable
    },
    return_source_documents=True  # Optional: Useful for debugging
)


# Streamlit app
st.subheader("NHS Health Assistant")
input_query = st.text_input("Ask a health-related question:")



if input_query:
    with st.spinner("Fetching response..."):
        try:
            # Correctly call the qa_chain with the expected key
            result = qa_chain.run(query=input_query)  # Use `.run()` for simplicity
            st.write(result)  # Directly display the result
        except ValueError as e:
            st.write(f"Input Query: {input_query}")
            st.write(f"Chain Config: {qa_chain}")
            st.error(f"An error occurred: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
