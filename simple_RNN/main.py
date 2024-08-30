import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# Set environment variable to disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Reset default graph (if using TensorFlow 1.x functions)
tf.compat.v1.reset_default_graph()

try:
    # Load the model
    model = load_model("Simple RNN_imdb.keras")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Proceed with the rest of your code


###Mapping of word index back to words (for our understanding)

word_index=imdb.get_word_index()
# word_index
reverse_word_index={value:key for key, value in word_index.items()}
model=load_model("Simple RNN_imdb.keras")
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Function to decode the reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?')for i in encoded_review])

#function to process the user input
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

### prediction function

def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment,prediction[0][0]

import streamlit as st
## streamlit app
#Streamlit app
st.title('IMDB movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

#User input

user_input=st.text_area(" Movie Review")

if st.button('Classify'):
    preprocess_input=preprocess_text(user_input)

    ##Make prediction

    prediction=model.predict(preprocess_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'

    ##Display the result
    st.write('Sentiment:',sentiment)
    st.write(f'Prediction Score:{prediction[0][0]}')
else:
    st.write('Please enter a movie review.')


 