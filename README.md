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

2. Session State: Session_state is a object that exists in the memory for us to use. We use it in this project to store user queries, system responses and our chat history.

3. Memory: Initializes a memory buffer gotten from langchain (`ConversationBufferWindowMemory()`) to store conversation history.

4. Templates: Creates prompt templates for system and human messages that guide the conversation flow.

5. Conversation Chain: The chain is create using langchains `ConversationChain()` class. Which sets up a system (chain of sequences) to manage the conversation, generating responses based on user input and maintain history. 

6. User Interaction and Response Generation: A text input field for users to enter queries. When submitted, the system generates a response and displays it along with the user query. 

