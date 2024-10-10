import pandas as pd
from tqdm.notebook import tqdm
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
from transformers import TextClassificationPipeline
nltk.download('punkt')
nltk.download('stopwords')

# reading dataset
df = pd.read_csv('Symptom2Disease.csv')

# making labels dict so that we can map the output of the model with the labels
int2label = {}

for i, disease in enumerate(df['label'].unique()):
    int2label[i] = disease

label2int = {v : k for k, v in int2label.items()}
num_classes = len(int2label)

#set of English stop words
stop_words = set(stopwords.words('english'))


def clean_text(sent):
    #remove punctuations
    sent = sent.translate(str.maketrans('','',string.punctuation)).strip()
    
    #remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(sent)
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words).lower()

# apply clean_text on text column of df
df["text"] = df["text"].apply(clean_text)

df['label'] = df['label'].map(lambda x : label2int[x])
X, y = df['text'].values, df['label'].values

# tokenizing the data
x_tokenizer = Tokenizer(filters = '')
x_tokenizer.fit_on_texts(X)
x_vocab = len(x_tokenizer.word_index) + 1

# splitting the data
train_x, val_x, train_y, val_y = train_test_split(X, y, test_size = 0.1, stratify = y)
train_x.shape, val_x.shape, train_y.shape, val_y.shape


BATCH_SIZE = 8

# changing data to tf
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
train_encodings = tokenizer(list(train_x), padding="max_length", truncation=True)
val_encodings = tokenizer(list(val_x), padding="max_length", truncation=True)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_y
)).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_y
)).batch(BATCH_SIZE)


num_classes = 24

# preparing model and compling it
model = TFAutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased", 
    num_labels = num_classes, 
    id2label = int2label, 
    label2id = label2int,
    output_attentions = True)

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(learning_rate = 3e-5),
    metrics = ['accuracy'])

EPOCHS = 3

history = model.fit(train_dataset, 
        epochs = EPOCHS, 
        validation_data = val_dataset)


# Initialize the pipeline
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=num_classes)

# Make predictions
pred1 = pipe("I am experiencing rashes on my skin. It is itchy and is now swelling. Even my skin is starting to peel.")
pred2 = pipe("I have constipation and belly pain, and it's been really uncomfortable. The belly pain has been getting worse and is starting to affect my daily life. Moreover, I get chills every night, followed by a mild fever.")

# print(pred1[0][0]['label'])

# # Print the highest label for each prediction
print("pred1:", pred1[0][0]['label'])
print("pred2:", pred2[0][0]['label'])

model.save_pretrained('bert_model')
tokenizer.save_pretrained('bert_tokenizer')