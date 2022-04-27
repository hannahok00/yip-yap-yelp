from sre_parse import Tokenizer
import pandas as pd
import numpy as np
import pickle
import nltk
import re
import string
import torch
import tensorflow
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from torch.nn.utils.rnn import pad_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


#In order to preprocess data we have a number of things we want to do:
# 1. remove punctuation
# 2. remove stop_words (common words)
# 3. stem - get the stem of words
# 4. lemmatize = 

def get_labels_data():
    #Read from the csv file to create a pandas dataframe
    data = pd.read_csv('../data/yelp_FastFood_dataset1.csv')

    #get the labels - corresponds to number of stars 
    labels = data['review_stars'].values
    reviews = data['text']
    #print(reviews[0:10])
    
    #Get's rid of all punctuation
    reviews = data['text'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '' , x))

    return labels[0:80000], reviews[0:80000]
    
#Function only for use if classifying as positive or negative
def classify_label(labels, cutoff):

    labels_list= []
    for label in labels:
        if label >= cutoff:
            labels_list.append(0)
        else:
            labels_list.append(1)
    #print(labels)
    return np.array(labels_list)

def tokenize(reviews):
    #Get the stop words 
    #A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) 
    #that a search engine has been programmed to ignore, both when indexing 
    # entries for searching and when retrieving them as the result of a search query. 
    stop_words = set(stopwords.words('english'))
    filtered_reviews = []

    for review in reviews:
        #print("tokenizing")
        #Tokenize the input
        tokenized_review = word_tokenize(review)

        #Remove the stop words
        filtered_sentence = [w for w in tokenized_review if not w.lower() in stop_words]

        #Stem the sentence
        ps = PorterStemmer()
        #stemmed_sentence = [ps.stem(w) for w in filtered_sentence]

        #Lemmatize the sentence
        lemmatizer = WordNetLemmatizer()
        #lemmatized_sentence = [lemmatizer.lemmatize(w) for w in stemmed_sentence]

        #Creates some funky bugs - missing trailing e, missing other elements
        filtered = [lemmatizer.lemmatize(w) if lemmatizer.lemmatize(w).endswith('e') else ps.stem(w) for w in filtered_sentence]
        filtered_reviews.append(filtered)

    #print(filtered_reviews)
    return filtered_reviews

def string_to_integer(reviews):
    # dictionary that maps integer to its string value 
    tokens_dict = {}

    # list to store integer labels 
    int_tokens = []

    for i in range(len(reviews)):
        tokens_dict[i] = reviews[i]
       # int_labels.append(i)

def pad_tokens(tokens):
    #tokens = [torch.tensor(w) for w in tokens]
    padded_tokens = pad_sequences(tokens, maxlen=50) 

    #print(padded_tokens)
    return padded_tokens


#def build_vocab(sentences):
#    tokens = []
#    all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN] + tokens)))
#    for s in sentences: tokens.extend(s)
#    vocab =  {word:i for i,word in enumerate(all_words)}

#	return vocab,vocab[PAD_TOKEN]


#def convert_to_id(vocab, sentences):
#    return np.stack([[vocab[word] if word in vocab else vocab[0] for word in sentence] for sentence in sentences])

def fit_text(reviews):
    t = Tokenizer()
    t.fit_on_texts(reviews)
    tokenized_words = t.texts_to_sequences(reviews)
    #print(tokenized_words)
    return tokenized_words

def preprocess():

    labels, reviews = get_labels_data()
    
    classified_labels = classify_label(labels, 3)
    reviews = tokenize(reviews)
    tokenized_words = fit_text(reviews)
    padded_tokens = pad_tokens(tokenized_words)

    train_inputs = padded_tokens[:64000]
    test_inputs = padded_tokens[64000:]
    train_labels = classified_labels[:64000]
    test_labels = classified_labels[64000:]

    return train_inputs, test_inputs, train_labels, test_labels




