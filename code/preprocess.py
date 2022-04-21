import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import max
from torch.nn import LSTM, Linear, Dropout, MaxPool1D, GRU, Conv1d, Emedding, Sequential

with open("file path goes here", "rb") as input_file:
    tokenizer = pickle.load(input_file)

####
data = pd.read_csv('dataset/yelp_FastFood_dataset.csv')
#DIRECTLY COPIED AND PASTED EXCEPT FOR FILEPATH
####

#####
# Selecting the reviews and ratings
X = data['text'].values
y = data['review_stars'].values
#DIRECTLY COPIED AND PASTED
######

###
cleaned_X = []
###

####
#DIRECTLY COPIED AND PASTED
# Iterating through the reviews
for i in range(len(X)):
    if i % 1000 == 0:
        print("--Cleaning {}th review--".format(i))
    cleaned_X.append(clean_review(X[i]))

# Converting the reviews into sequence
X_vec = tokenizer.texts_to_sequences(X)

# Padding the reviews
X_vec_pad = pad_sequences(X_vec, MAX_WORDS, padding='post')

# Stacking the data
dataset = np.hstack((X_vec_pad, y.reshape(-1,1)))

# Saving the dataset as numpy file
np.save('data/yelp_FastFood_dataset', dataset)
#DIRECTLY COPIED
####