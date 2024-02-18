# Python-Bot

## Table of Contents

- [About](#about)
- [Requirements](#requirements)
- [Project Setup](#project-setup)
- [How It Works](#how-it-works)
- [Run Streamlit App](#run-streamlit-app)
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

2. Session State: Session_state is a streamlit object that exists in the memory for retrieval. We use it in this project to store user queries, system responses and our chat history.

3. Memory: We Initialize [ConversationBufferWindowMemory](https://python.langchain.com/docs/modules/memory/types/buffer_window) a langchain library that acts as a form of keeping memory. It keeps a list of the interactions of conversations over a selected window in a buffer and passes them to our prompts.

4. Chat Prompts: Chat Prompts are instructions for the system that guide the the structure and content of the generated system message prompts. Prompts take in system and user templates. System templates determine the structure and content of the generated system message prompts. Human templates indicates that the template expects the user to provide some input, and this input will be incorporated into the generated prompts.

5. Conversation Chain: The chain (A chain represents a sequence of system responses and user queries) is created using langchains [ConversationChain](https://python.langchain.com/docs/modules/memory/conversational_customization) library. Which sets up a system to manage the conversation, generating responses based on user input and maintain history. It takes the memory, chat prompts and a specified Large Language Model.

6. User Interaction and Response Generation: There is a text input field generated by streamlit for users to enter questions. When submitted, The questions are initialised as an input in the `predict()` function from the `ConversationChain()` library. What this does is pass the questions to the initialised Large Language model in the `ConversationChain()` constructor. A response is then generated based on the prompts set earlier and displayed in the user interface. 

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
