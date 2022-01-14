import json 
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle
import time

from tensorflow.python.keras.layers.core import Dropout

with open('/Users/sourya/Desktop/Chatbot project/intents.json') as file:
    data = json.load(file)
    
training_sentences = []
training_labels = []
labels = []
responses = []


for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])
    
    if intent['tag'] not in labels:
        labels.append(intent['tag'])
        
num_classes = len(labels)

lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)

vocab_size = 2000
embedding_size = 40
max_len = 50

tokenizer = Tokenizer(num_words = vocab_size, oov_token = "<OOV>")
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating = 'pre', maxlen = max_len)

model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length = max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(40, activation = "relu"))
model.add(Dense(40, activation = "relu"))
model.add(Dense(num_classes, activation = "softmax"))

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.summary()

trained_model = model.fit(padded_sequences, np.array(training_labels), epochs = 300, verbose = 2)

model.save("chatbot_model.h5")

pickle.dump(tokenizer, open("tokenizer.pkl",'wb'))
pickle.dump(lbl_encoder,open("label_encoder.pkl",'wb'))
