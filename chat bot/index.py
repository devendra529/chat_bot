import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import random
from flask import Flask, render_template, request, jsonify

nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

training_sentences = [
    "Hi", "Hello", "Hey", "How are you?", "What's up?", "Good morning", 
    "Good night", "Tell me a joke", "What is your name?", "Where are you from?", 
    "Tell me a story", "How old are you?", "What is the weather today?", 
    "How do I make a cup of tea?", "What is Python?", "Who are you?", 
    "Tell me a fun fact", "Can you help me?", "Bye", "Quit","What is your favorite color?", "Do you like movies?", "Tell me a fun fact about space", 
    "How do I bake a cake?", "What is AI?", "Can you sing?", "What is your favorite food?", 
    "Do you believe in love?", "What is the meaning of life?", "Do you like reading books?", 
    "What's your favorite song?", "Can you recommend a good movie?", "What time is it?", 
    "Can you play games?", "What’s your favorite hobby?", "Can you speak other languages?", 
    "Tell me a riddle", "Are you real?", "Can you teach me something?", "Do you know any jokes about cats?"
]

training_labels = [
    "greeting", "joke", "name", "location", "story", "age", "weather", 
    "how_to_make_tea", "python_info", "who", "fun_fact", "help", "goodbye", "goodbye",
     "help", "goodbye", "goodbye" "help", "goodbye", "goodbye","goodbye","favorite_color", "movies", "space_fact", "how_to_bake_cake", "AI_info", "singing", 
    "favorite_food", "believe_in_love", "meaning_of_life", "book_lovers", "favorite_song", 
    "movie_recommendation", "time", "play_games", "hobbies", "languages", "riddle", 
    "real_or_not", "teach_me", "cat_jokes"
]

def generate_more_sentences(num_sentences=10000):
    greetings = ["Hi", "Hello", "Hey", "How are you?", "What's up?", "Good morning", "Good night", "Yo", "Howdy"]
    jokes = [
       "tell me jokes", "some jokes", "tell me some jokes",
    ]
    names = ["What is your name?", "Can you tell me your name?", "Who are you?", "What's your name?"]
    location = ["Where are you from?", "Where do you live?", "Where are you based?", "What's your location?"]
    stories = ["Tell me a story", "Can you share a story?", "Once upon a time", "Give me a fun story"]
    ages = ["How old are you?", "What’s your age?", "How many years old are you?", "What’s your birth year?"]
    weather = ["What is the weather like?", "How’s the weather?", "What’s the weather today?", "Is it going to rain today?"]
    how_to_make_tea = ["How do I make a cup of tea?", "Can you show me how to make tea?", "What are the steps to make tea?"]
    python_info = ["What is Python?", "Tell me about Python", "Can you explain Python?", "What does Python do?"]
    who = ["Who are you?", "What are you?", "Tell me about yourself", "What is your purpose?"]
    fun_facts = ["Tell me a fun fact", "Share an interesting fact", "I want to know something fun", "Give me a cool fact"]
    help = ["Can you help me?", "I need help", "Can you assist me?", "Please help me"]
    goodbyes = ["Bye", "Goodbye", "See you later", "Take care", "Catch you later", "Farewell"]
    favorite_color =["What is your favorite color?"]

    categories = {
        "greeting": greetings,
        "joke": jokes,
        "name": names,
        "location": location,
        "story": stories,
        "age": ages,
        "weather": weather,
        "how_to_make_tea": how_to_make_tea,
        "python_info": python_info,
        "who": who,
        "fun_fact": fun_facts,
        "help": help,
        "goodbye": goodbyes,
        
    }

    sentences = []
    labels = []
    
    for _ in range(num_sentences):
        intent = random.choice(list(categories.keys()))
        sentence = random.choice(categories[intent])
        sentences.append(sentence)
        labels.append(intent)
   
    assert len(sentences) == len(labels), "Mismatch between sentences and labels length!"

    return sentences, labels
    print(f"Training sentences length: {len(training_sentences)}")
print(f"Training labels length: {len(training_labels)}")


generated_sentences, generated_labels = generate_more_sentences(100000)

training_sentences.extend(generated_sentences)
training_labels.extend(generated_labels)

def preprocess_sentence(sentence):
    return " ".join([lemmatizer.lemmatize(word.lower()) for word in sentence.split()])

model = make_pipeline(TfidfVectorizer(stop_words='english'), LogisticRegression(max_iter=200))

model.fit(training_sentences, training_labels)

def chatbot_response(user_input):
    
    predicted_label = model.predict([user_input])[0]
    
    print(f"Predicted intent: {predicted_label}")
    
    responses = {
        "greeting": [
            "Hello! How's it going?", 
            "Hey there! What's up?", 
            "Hi! How can I help you today?"
        ],
        "joke": [
            "Why don’t skeletons fight each other? They don’t have the guts!",
            "What do you call fake spaghetti? An impasta!",
            "Why don’t some couples go to the gym? Because some relationships don’t work out!"
        ],
        "name": [
            "I am Janu, your friendly assistant!", 
            "Call me Janu. I'm here to chat with you!"
        ],
        "location": [
            "I live in Mumbai, India. How about you?",
            "I’m from the beautiful city of Mumbai in India."
        ],
        "story": [
            "Once upon a time, in a land far, far away, a little bird learned to fly...",
            "A young adventurer traveled across distant lands in search of treasure, but found something even more valuable..."
        ],
        "age": [
            "I’m quite young, still learning new things every day!",
            "I was born just yesterday in the world of artificial intelligence, so I’m ageless!"
        ],
        "weather": [
            "I can't tell you the weather, but you can easily check it on your phone!",
            "You can get up-to-date weather info on your phone or laptop!"
        ],
        "how_to_make_tea": [
            "First, heat some water, then add your tea leaves or tea bag, and let it steep for a few minutes!",
            "You boil some water, add tea leaves or a tea bag, and let it steep until it’s just right."
        ],
        "python_info": [
            "Python is an amazing programming language that’s simple to learn and super versatile!",
            "Python is a high-level programming language. It’s great for everything from web development to data analysis!"
        ],
        "who": [
            "I’m dev, here to chat and help you with anything you need!",
            "I’m a friendly bot named dev. I’m here to assist you in any way I can!"
        ],
        "fun_fact": [
            "Did you know? Honey never spoils. Archaeologists have found pots of honey in ancient tombs that are still edible!",
            "Here’s a fun fact: Bananas are berries, but strawberries aren’t!"
        ],
        "help": [
            "Sure, I’m happy to help! What do you need?",
            "I’m here to assist you with anything! Just let me know what you need help with."
        ],
        "goodbye": [
            "Goodbye! Come back anytime!",
            "It was nice chatting with you. Take care!"
        ]
    }
 
    return random.choice(responses.get(predicted_label, ["Sorry, I didn't quite understand that. Can you rephrase?"]))



# Flask web app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json.get('user_input')
    
    if not user_input:
        return jsonify({'response': "Please type something!"})
   
    user_input = preprocess_sentence(user_input)
    response = chatbot_response(user_input)
    
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)

import pandas as pd

df = pd.read_csv('additional_training_data.csv')

training_sentences.extend(df['sentence'].tolist())
training_labels.extend(df['label'].tolist())
import json

with open('additional_training_data.json') as f:
    new_data = json.load(f)

for item in new_data:
    training_sentences.append(item['sentence'])
    training_labels.append(item['label'])
import requests

response = requests.get("https://api.openweathermap.org/data/2.5/weather?q=London&appid=YOUR_API_KEY")
weather_data = response.json()

weather_question = f"What is the weather like in {weather_data['name']}?"
training_sentences.append(weather_question)
training_labels.append('weather')
from nltk.corpus import wordnet

def generate_synonym(sentence):
    words = sentence.split()
    augmented_sentence = []
    
    for word in words:
        synsets = wordnet.synsets(word)
        if synsets:
            synonyms = set()
            for syn in synsets:
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name())
            if synonyms:
                augmented_sentence.append(synonyms.pop())  
            else:
                augmented_sentence.append(word)
        else:
            augmented_sentence.append(word)
    
    return ' '.join(augmented_sentence)


augmented_sentences = [generate_synonym(sentence) for sentence in training_sentences]
training_sentences.extend(augmented_sentences)
training_labels.extend(training_labels) 
