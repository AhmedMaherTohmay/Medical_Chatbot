# Import necessary librariesC
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
import pandas as pd

# Load the pre-trained BERT model and tokenizer
model = TFAutoModelForSequenceClassification.from_pretrained("bert_model\Saved_model/")
tokenizer = AutoTokenizer.from_pretrained("bert_model\Saved_tokenizer/")

# Load the CSV file containing symptom-to-disease mappings
df = pd.read_csv('bert_model\Symptom2Disease.csv')

def return_label(prediction):
    """
    Convert the prediction index to the corresponding disease label.
    """
    int2label = {}
    for i, disease in enumerate(df['label'].unique()):
        int2label[i] = disease
    return int2label.get(prediction)

# Create a mapping from disease labels to doctors
doctor_mapping = dict(zip(df['label'], df['doctors']))

def make_prediction(text, threshold=0.5):
    """
    Make a prediction for the given text using the BERT model.
    If the maximum probability is below the threshold, return 'Unknown'.
    """
    encodings = tokenizer(text, padding="max_length", truncation=True, return_tensors="tf")
    outputs = model(encodings)
    logits = outputs.logits
    probabilities = tf.nn.softmax(logits, axis=-1)
    max_prob = tf.reduce_max(probabilities, axis=-1).numpy()
    if max_prob[0] < threshold:
        return "Unknown"
    
    predictions = tf.argmax(logits, axis=-1).numpy()
    return predictions[0]

def bert_response(symptom):
    """
    Generate a response based on the given symptom.
    """
    pre_ans = "Based on these symptoms, the most likely disease is "
    aft_ans = ", and I suggest you go see a "
    prediction = make_prediction(symptom)
    if prediction == 'Unknown':
        return "Unknown"
    else:
        disease = return_label(prediction)
        doctor = doctor_mapping.get(disease, "Doctor not found")
        return pre_ans + disease + aft_ans + doctor

# Example usage
if __name__ == '__main__':
    symptom = "I noticed a sudden onset of high fever, along with a rash on my skin and severe pain behind my eyes."
    bert = bert_response(symptom)
    print(bert)
