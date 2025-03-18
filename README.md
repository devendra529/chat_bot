# chat_bot
This project is a simple Chatbot built using Python, Flask, and Machine Learning models like Logistic Regression

# Features
Intent Recognition: The chatbot recognizes various intents like greetings, asking for weather, telling jokes, etc.
Natural Language Processing (NLP): It uses nltk for text preprocessing and intent classification.
Machine Learning Model: The chatbot uses Logistic Regression for predicting the user's intent based on input text.
Flask Web App: The chatbot is deployed on a simple web application built with Flask.
Multiple Intents: It can handle various queries such as greetings, jokes, time, favorite colors, and more.

# Technologies Used
Python: The primary language for implementing the chatbot logic.
Flask: A micro web framework to create the chatbot web app.
nltk: Used for text preprocessing, tokenization, and lemmatization.
HTML/CSS: For creating the frontend user interface.
JSON: For storing training data (user inputs and responses).

# Installation
Follow these steps to get the chatbot up and running locally:
# 1. Clone the Repository
git clone https://github.com/devendra529/chat_bot.git
cd chat_bot

# 2. Install Dependencies
Install all necessary libraries using pip:

pip install -r requirements.txt
The requirements.txt file includes all the required libraries like:

Flask
scikit-learn
nltk
pandas
requests

# Example Interactions
Here are some example queries you can ask the chatbot:

"Hi"
"Tell me a joke"
"What is your name?"
"Where are you from?"
"What is the weather like?"
"Can you help me?"
"Goodbye"
