# Python-Bot

## Table of Contents

- [About](#about)
- [Requirements](#requirements)
- [Project Setup](#project-setup)
- [How It Works](#how-it-works)
- [Run Streamlit App](#run-streamlit-app)
- [Usage](#usage)
- [Improvements](#improvements)

## About

This is python expert chatbot who converses and shows you code snippets on any python problem you face.

The project was built using [langchain](https://python.langchain.com/docs/get_started/introduction) to create the chatbot and [streamlit](https://streamlit.io/) for the UI

## Requirements

You need python installed to run this project

## Project Setup

To get started with this project, follow these steps:

Clone the repository to your local machine:

```bash
git clone https://github.com/gbemike/Python-Bot.git
```

Navigate to the project directory:

```bash
cd Python-Bot
```

## How It Works

The application follows these steps to provide responses to your questions:

1. Setup: We first import necessary libraries then retrieve our OpenAI API key. You'll need to load your API_KEY to use this bot.

2. Session State: Session_state is a streamlit object that exists in the memory for us to use. We use it in this project to store user queries, system responses and our chat history.

3. Memory: We Initialize a [ConversationBufferWindowMemory](https://python.langchain.com/docs/modules/memory/types/buffer_window) to store conversation history. It is a Langchain library that keeps a list of the interactions of the conversation over a selected window.

4. Chat Prompts: Chat Prompts are instructions for the system that guide the the structure and content of the generated system message prompts. Prompts take in system and user templates. System templates determine the structure and content of the generated system message prompts. Human templates indicates that the template expects the user to provide some input, and this input will be incorporated into the generated prompts.

5. Conversation Chain: The chain is created using langchains [ConversationChain](https://python.langchain.com/docs/modules/memory/conversational_customization) library. Which sets up a system (chain of sequences) to manage the conversation, generating responses based on user input and maintain history. 

6. User Interaction and Response Generation: A text input field for users to enter queries. When submitted, the system generates a response and displays it along with the user query. 

## Run Streamlit App

1. Navigate to project directory

```bash
cd Python-Bot
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run sreamlit app

```bash
streamlit run app.py
```

4. The application will launch in your defualt web browser, displaying the user interface.

5. Asks questions relating to python in the chat interface
