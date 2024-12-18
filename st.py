import random
import json
import numpy as np
import nltk
import pickle
import streamlit as st
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

# Initialize necessary components
lemmatizer = WordNetLemmatizer()

# Load model and other necessary files
model = load_model("chatbot_model.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

# Function to clean up the sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to create bag of words
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

# Predict class function
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Get the response from intents file
def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

# Streamlit UI
st.set_page_config(page_title="Career Guidance ChatBot", page_icon=":robot_face:", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        .chat-container {
            background-color: #F9F9F9;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
        }

        .message {
            padding: 10px 15px;
            border-radius: 20px;
            margin-bottom: 10px;
            max-width: 70%;
        }

        .user-message {
            background-color: #0078D4;
            color: white;
            float: right;
            clear: both;
            display: inline-block;
            border-radius: 20px;
            max-width: 80%;
        }

        .bot-message {
            background-color: #E0E0E0;
            color: black;
            float: left;
            clear: both;
            display: inline-block;
            border-radius: 20px;
            max-width: 80%;
        }

        .input-container {
            margin-top: 20px;
        }

        .chatbox {
            overflow-y: auto;
            max-height: 400px;
            padding: 10px;
            background-color: #fff;
            border-radius: 15px;
            margin-bottom: 20px;
        }

        .input-group {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .input-field {
            width: 85%;
            padding: 12px;
            font-size: 16px;
            border-radius: 20px;
            border: 1px solid #ccc;
        }

        .send-button {
            width: 10%;
            background-color: #0078D4;
            color: white;
            padding: 12px;
            border-radius: 20px;
            border: none;
            cursor: pointer;
        }

        .send-button:hover {
            background-color: #005A9E;
        }

        .icon {
            font-size: 24px;
            margin-right: 10px;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for user input
user_input = st.sidebar.text_input("Type your message:", "")

# Chat history display
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "bot", "text": "Hi! I'm your chatbot. How can I assist you today?"}]

# Function to display the chat history
def display_chat():
    chat_box = st.container()
    with chat_box:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="message user-message"><i class="fas fa-user icon"></i><span>{message["text"]}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="message bot-message"><i class="fas fa-robot icon"></i><span>{message["text"]}</span></div>', unsafe_allow_html=True)

# Handle user input and chatbot response
if user_input:
    if user_input.lower().startswith(('my name is', 'hi my name is')):
        name = user_input.split('is')[1].strip()
        ints = predict_class(user_input, model)
        response = getResponse(ints, intents)
        response = response.replace("{n}", name)
    else:
        ints = predict_class(user_input, model)
        response = getResponse(ints, intents)
    
    # Save user input and bot response to session state
    st.session_state.messages.append({"role": "user", "text": user_input})
    st.session_state.messages.append({"role": "bot", "text": response})

# Display chat history
display_chat()






# import random
# import json
# import numpy as np
# import nltk
# import pickle
# import streamlit as st
# from keras.models import load_model
# from nltk.stem import WordNetLemmatizer

# # Initialize necessary components
# lemmatizer = WordNetLemmatizer()

# # Load model and other necessary files
# model = load_model("chatbot_model.h5")
# intents = json.loads(open("intents.json").read())
# words = pickle.load(open("words.pkl", "rb"))
# classes = pickle.load(open("classes.pkl", "rb"))

# # Function to clean up the sentence
# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# # Function to create bag of words
# def bow(sentence, words, show_details=True):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for s in sentence_words:
#         for i, w in enumerate(words):
#             if w == s:
#                 bag[i] = 1
#                 if show_details:
#                     print("found in bag: %s" % w)
#     return np.array(bag)

# # Predict class function
# def predict_class(sentence, model):
#     p = bow(sentence, words, show_details=False)
#     res = model.predict(np.array([p]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#     results.sort(key=lambda x: x[1], reverse=True)
#     return_list = []
#     for r in results:
#         return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
#     return return_list

# # Get the response from intents file
# def getResponse(ints, intents_json):
#     tag = ints[0]["intent"]
#     list_of_intents = intents_json["intents"]
#     for i in list_of_intents:
#         if i["tag"] == tag:
#             result = random.choice(i["responses"])
#             break
#     return result

# # Streamlit UI
# st.title("Career Guidance ChatBot")

# # Create a text input box for the user
# user_input = st.text_input("Type your message:")

# # Display the bot's reply
# if user_input:
#     if user_input.lower().startswith(('my name is', 'hi my name is')):
#         name = user_input.split('is')[1].strip()
#         ints = predict_class(user_input, model)
#         response = getResponse(ints, intents)
#         response = response.replace("{n}", name)
#     else:
#         ints = predict_class(user_input, model)
#         response = getResponse(ints, intents)
    
#     st.markdown(f"**Bot:** {response}")

# # Display a default greeting message when no input is given
# else:
#     st.markdown("**Bot:** Hi! I'm your chatbot. How can I assist you today?")

