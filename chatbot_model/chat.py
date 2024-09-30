# Import necessary libraries
import random
import json
import torch
import re

# Import custom modules for the chatbot model and utility functions
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Set the device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the intents file
with open('chatbot_model/intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load the trained model data
FILE = "chatbot_model/data.pth"
data = torch.load(FILE)

# Extract model parameters and other necessary data
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize the model and load the saved state
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Define the bot's name
bot_name = "Innovatex"

def extract_name(msg):
    # Look for patterns such as "I am [name]", "My name is [name]", etc.
    name_pattern = re.compile(r"(?:my name is|i am|this is|it is)\s+([\w\s]+)", re.IGNORECASE)
    match = name_pattern.search(msg)
    if match:
        return match.group(1)
    return None

# Function to get a response from the chatbot
def get_response(msg):
    # Tokenize the input message
    sentence = tokenize(msg)
    user_name = extract_name(msg)
    # Convert the tokenized sentence to a bag-of-words vector
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Get the model's output
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    # Get the predicted tag
    tag = tags[predicted.item()]

    # Calculate the probabilities of each class
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    # If the probability is above a threshold, return a random response from the corresponding intent
    if prob.item() > 0.6:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                
                # Replace the <HUMAN> placeholder with the extracted user name
                if user_name:
                    print(user_name)
                    user_name = re.sub(r'[^a-zA-Z\s]', '', user_name)
                    response = response.replace("<HUMAN>!", user_name)
                else:
                    response = response.replace("<HUMAN>", "there")  # Default placeholder if no name is provided

                return response
    # If the probability is below the threshold, return a default response
    return "I do not understand..."

# Main function to run the chatbot in a loop
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # Get user input
        sentence = input("You: ")
        if sentence == "quit":
            break

        # Get the chatbot's response and print it
        resp = get_response(sentence)
        print(resp)
