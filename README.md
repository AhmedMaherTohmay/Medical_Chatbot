# AI Team

This repository contains the AI components for a medical chatbot that helps patients detect their diseases based on given symptoms. The chatbot combines two models: one for friendly chat, created using a simple neural network (NN), and another for predicting diseases, built using a pretrained BERT model.

## Deployment Options

You have two deployment options:

* **Use Docker.**
* **Use python anywhere by uploading your files and running the app**.

## Repository Structure

This repository is divided into two parts:

## First part is for Chatbot Model

### The Chatbot model is made by two models and integrated with front end

### 1. BERT Model

#### Setup

1. **Clone the repository and create a virtual environment:**
   ```bash
   $ git clone https://github.com/InnovateX-Team/Machine-leaning.git
   $ cd chatbot-deployment
   $ python3 -m venv venv
   $ . venv/bin/activate
   ```
2. **Install dependencies:**
   ```bash
   $ (venv) pip install Flask torch torchvision nltk transformers
   ```
3. **Install NLTK package:**
   ```bash
   $ (venv) python
   >>> import nltk
   >>> nltk.download('punkt')
   ```

#### Training and Testing

* **Run the Notebook** you will find the notebooke for the model in bert model folder. Just run it in kaggle and download the model and it's tokenizer to use it 
* **Test the BERT model:**

```bash
  $ (venv) python bert_model.py
```

### 2. Chatbot Model

#### Setup

* **Modify `intents.json`** with different intents and responses for your chatbot.
* In case you don't want to change anything, then test the model as I already saved the model

#### Training and Testing

* **Train the model:**

  ```bash
  $ (venv) python train.py
  ```

  This will generate a `data.pth` file.
* **Test the model in the console:**

  ```bash
  $ (venv) python chat.py
  ```

### 3. Frontend Integration

* **Static and Templates Folders:**
  * These folders contain the JavaScript, CSS, and HTML files used for this app.

## Deployment

* **Run the Flask app:**

```bash
  $ (venv) python app.py
```
  This command runs the main file, which contains the Flask API to run the app.


## Using Docker
1. Build the Docker image:
 
 ```bash
  docker build -t chatbot-app .

```

2. Run the Docker container:
 
 ```bash
 docker run -d -p 5000:5000 chatbot-app

```

## Note

I added a requriments.txt that contains all the dependencies needed to run the program

I also added comments in the code to explain it in more details
