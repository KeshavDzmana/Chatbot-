# Chatbot-
I have created this project during my data science course 
import random
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download necessary NLTK resources

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Hyperparameters (adjust as needed)

MAX_SENTENCE_LENGTH = 20  # Maximum length of sentences
MAX_NUM_RESPONSES = 5  # Maximum number of responses to generate

# Data preprocessing functions

def preprocess_sentence(sentence):
    """
    Preprocesses a sentence:
    - Tokenizes sentence
    - Lowercases words
    - Removes punctuation
    - Lemmatizes words
    """
    tokens = nltk.word_tokenize(sentence)
    tokens = [word.lower() for word in tokens if word.isalnum()]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens[:MAX_SENTENCE_LENGTH]

def load_training_data(filename):
    """
    Loads training data from a file:
    - Reads lines from the file
    - Creates a list of conversation pairs (question, answer)
    """
    conversations = []
    with open(filename, 'r') as f:
        for line in f:
            question, answer = line.strip().split('|')
            conversations.append((preprocess_sentence(question), preprocess_sentence(answer)))
    return conversations

# for Model training

def train_model(conversations):
    """
    Trains a Naive Bayes classifier on the conversation data:
    - Creates a vectorizer to represent sentences as numerical features
    - Trains a MultinomialNB classifier to predict responses
    """
    vectorizer = CountVectorizer()
    x_train = vectorizer.fit_transform([conversation[0] for conversation in conversations])
    y_train = [conversation[1] for conversation in conversations]
    model = MultinomialNB()
    model.fit(x_train, y_train)
    return model, vectorizer
    
def generate_response(model, vectorizer, sentence, max_num_responses):
    """
    Generates a response to a sentence using the trained model:
    - Preprocesses the sentence
    - Converts it to features using the vectorizer
    - Predicts the most likely responses from the model
    - Selects a random response from the top predictions
    """
    preprocessed_sentence = preprocess_sentence(sentence)
    sentence_features = vectorizer.transform([preprocessed_sentence])
    predicted_indices = model.predict_proba(sentence_features)[0].argsort()[-max_num_responses:]
    responses = [vectorizer.get_feature_names()[i] for i in predicted_indices]
    return random.choice(responses)
if __name__ == '__main__':
    conversations = load_training_data('chatbot_data.txt')
    model, vectorizer = train_model(conversations)

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = generate_response(model, vectorizer, user_input, MAX_NUM_RESPONSES)
        print("Bot:", response)
        
