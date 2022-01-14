import json 
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import random
import pickle


tokenizer = pickle.load(open("/Users/sourya/Desktop/Chatbot project/tokenizer.pkl",'rb'))
lbl_encoder = pickle.load(open("/Users/sourya/Desktop/Chatbot project/label_encoder.pkl",'rb'))
data = json.load(open("/Users/sourya/Desktop/Chatbot project/intents.json"))
model = keras.models.load_model("/Users/sourya/Desktop/Chatbot project/chatbot_model.h5")

bot_name = "bot"

def predict_tag(input_sentence):

    tokenized_input_sequences = tokenizer.texts_to_sequences([input_sentence])

    padded_input_sentence = pad_sequences(tokenized_input_sequences, maxlen = 50, truncating = 'pre')

    result = model.predict(padded_input_sentence)

    output = max(result[0])

    if (float(output) == 0.9443353414535522):

        tag = 0

    else:

        tag = lbl_encoder.inverse_transform([np.argmax(result)])

    return tag

def response(tag, data_json):

    if (tag == 0):

        result = "Sorry,I don't understand."

    else:

        for i in data_json["intents"]:

            if (i["tag"] == tag):

                result = random.choice(i["responses"])
                break

    return result

def chatbot_response(input_sentence):

    result = predict_tag(input_sentence)
    output = response(result, data)
    return output