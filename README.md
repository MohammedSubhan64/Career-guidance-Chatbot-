# Career Guidance ChatBot
An AI Chatbot using Python and Flask 

## Project Description
This project creates an intelligent chatbot using Flask, Keras, and Natural Language Processing (NLP) techniques. The chatbot classifies user input, predicts intents, and generates appropriate responses based on predefined intent patterns. The system includes a web interface where users can interact with the bot in real-time.

## Key Features:
Intent Prediction: The chatbot classifies user input by predicting the intent using a trained machine learning model built with Keras.
Personalized Responses: If the user mentions their name, the bot personalizes its responses.
Natural Language Processing: The chatbot processes input text using tokenization, lemmatization, and a bag-of-words model to improve its understanding of the message.
Model Training: The chatbot model is trained on labeled data consisting of patterns and intents, using deep learning techniques to predict the appropriate response.
Web Interface: Users interact with the chatbot through a simple web form where they submit their queries.

## Technologies Used:
Flask: For building the backend of the web application.
Keras (TensorFlow): For loading and training the deep learning model.
nltk: For text preprocessing (tokenization, lemmatization).
NumPy: For handling numerical data and array operations.
Pickle & JSON: For storing and loading the chatbot's vocabulary, classes, and intents.
SGD Optimizer: For training the model using the Stochastic Gradient Descent (SGD) algorithm.
Workflow:
Text Preprocessing: User input is tokenized and lemmatized to standardize words.
Bag of Words: The input is converted into a bag-of-words vector that represents the presence of words from the predefined vocabulary.
Intent Prediction: The trained model predicts the intent of the input message by comparing it to labeled patterns in the training data.
Response Selection: The chatbot fetches a random response from the list of responses associated with the predicted intent.
Personalization: If the user mentions their name, the bot inserts it into the response for a personalized touch.

## Training the Model:
The model is trained on a dataset of user patterns and corresponding intents from a JSON file. It uses a neural network with three layers: an input layer with 128 neurons, a hidden layer with 64 neurons, and an output layer with neurons equal to the number of intents.
The model is compiled using the Stochastic Gradient Descent (SGD) optimizer with Nesterov accelerated gradients to improve accuracy during training. The model is trained for 200 epochs with a batch size of 5.

## Use Cases:
Customer Support: Answering frequent customer queries automatically.
Personal Assistants: Providing interactive responses to basic user requests.
Education: Assisting in learning by answering subject-specific questions.

## Future Improvements:
Add context-awareness, multilingual support, and more advanced models.

