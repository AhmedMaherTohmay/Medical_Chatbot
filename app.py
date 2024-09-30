# Import necessary modules from Flask and custom models
from flask import Flask, request, jsonify, render_template
from flasgger import Swagger, swag_from
import nltk
nltk.download('punkt_tab')
from chatbot_model.chat import get_response
from bert_model.bert_model import bert_response
from flasgger import Swagger, swag_from

import nltk
nltk.download('punkt_tab')
# Initialize the Flask application
app = Flask(__name__)

# Initialize Swagger
swagger = Swagger(app)

# Define a route for the home page
@app.get("/")
def index_get():
    # Render the base HTML template when the home page is accessed
    return render_template('base.html')

# Define a route for the prediction endpoint
@app.post("/predict")
@swag_from({
    'responses': {
        200: {
            'description': 'A JSON response with the predicted answer',
            'examples': {
                'application/json': {
                    "answer": "This is the response message"
                }
            }
        }
    },
    'parameters': [
        {
            'name': 'message',
            'in': 'body',
            'type': 'string',
            'required': True,
            'description': 'The message text to predict a response for',
            'schema': {
                'type': 'object',
                'properties': {
                    'message': {
                        'type': 'string'
                    }
                },
                'example': {
                    'message': 'Hello, how are you?'
                }
            }
        }
    ]
})
def predict():
    # Get the JSON data from the request
    text = request.get_json('message')
    
    # Get the response from the BERT model
    bert = bert_response(text['message'])
    
    # If BERT model returns 'Unknown', use the chatbot model to get a response
    if bert == 'Unknown':
        response = get_response(text['message'])
        message = {"answer": response}
    else:
        # If BERT model returns a valid response, use it
        message = {"answer": bert}
        print(message)
    
    # Return the response as a JSON object
    return jsonify(message)

# Run the Flask application in debug mode
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
