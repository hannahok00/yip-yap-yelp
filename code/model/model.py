import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, sk_text
from torch import max
from torch.nn import LSTM, Linear, Dropout, MaxPool1D, GRU, Conv1d, Emedding, Sequential

#not sure of the equivalent to earlystopping or modelcheckpoint

#linear is equivalent to dense layers
#GRU layer would represent the bidirectional layer you can set GRU bidirectional = true 
#for global max pooling can do output, _ = torch.max(input, 1)



#here would need to read from the tokenized pickle file
with open("file name goes here", "rb") as input_file: 
    tokenzier = pickle.load(input_file)


#constants
GPU = False
MAX_WORDS = 80
NUMBER_OF_CLASSES = 5
VOCAB_SIZE = 10000
EPOCHS = 50
BATCH_SIZE = 1024

dataset = np.load('data set goes here')

X = dataset[:,80]
y = dataset[:,80] #i think represents stars
#yelp github uses 
#Tfidf_vectorizer.get_feature_names()) and then uses ['stars']
y = pd.get_dummies(y).values

del dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 1969)


del X, y 

model = Sequential()
model.add(Emedding(VOCAB_SIZE, 128, input_length=MAX_WORDS))
model.add(Dropout(0.5))
model.add(Conv1d(filters=256, kernel_size=3, input_length=MAX_WORDS))
model.add(MaxPool1D(3))
model.add(Conv1d(filters=512, kernel_size=3, activation='relu'))
model.add(MaxPool1D(3))
model.add(Dropout(0.5))
#did not add the if gru statement
model.add(LSTM(256, return_sequences=True))
#did not add global max pool -- pytorch has no such function 
model.add(Linear(256, activation='relu'))
model.add(Linear(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Linear(NUMBER_OF_CLASSES, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

#did not know how to implement early stopping and model checkpoint in pytorch/our code

##i feel as if we should do something other than the model.fit to train the data
#kind of confused by it


model.fit(X_train, 
          y_train, 
          validation_data=(X_test, y_test), 
          epochs=EPOCHS, 
          batch_size=BATCH_SIZE)


# Saving the model
model.save('data/model_best.h5') 
# bridget: I switched this from him, it was dataset/model_best.h5 but I am confused because ..
# when he saves it he does it under model file

