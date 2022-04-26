import pickle
from xml.etree.ElementPath import prepare_parent
import numpy as np
import pandas as pd
import torch
from torch import max
from torch import optim
from torch.nn import LSTM, Linear, Dropout, MaxPool1d, GRU, Conv1d, Embedding, Sequential, ReLU, Softmax, Sigmoid
from real_preprocess import preprocess

#not sure of the equivalent to earlystopping or modelcheckpoint

#linear is equivalent to dense layers
#GRU layer would represent the bidirectional layer you can set GRU bidirectional = true 
#for global max pooling can do output, _ = torch.max(input, 1)



#constants
GPU = False
MAX_WORDS = 20
#Depends on binary classification or not
NUMBER_OF_CLASSES = 2
#Idk what vocab size is 
VOCAB_SIZE = 10000
EPOCHS = 50
BATCH_SIZE = 1024

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    
        #self.model = Sequential()
        #Need to figure out vocab size
        self.embedding = Embedding(VOCAB_SIZE, 128)
        self.LSTM = LSTM(128, 300)
        self.l1 = Linear(300, 100)
        self.relu = ReLU()
        self.l2 = Linear(128, 2)
        self.softmax = Softmax()
    
        self.sigm = Sigmoid()
        #self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)

        self.loss_list = []

    def call(self, reviews):
        
        #Need to figure out vocab size
        #The shape of the self.embedding will be [sentence_length, batch_size, embedding_dim]
        l1_out = self.embedding(reviews)
        l2_out, hidden_state = self.LSTM(l1_out)
        #print(l2_out.shape)
        l3_out = self.l1(l2_out)
        l4_out = self.relu(l3_out)
        l5_out = self.l2(l4_out) 
        final_out = self.softmax(l5_out)
        return final_out

    def loss(self, labels, predictions):
        #Might need to reshape datat

        loss = torch.nn.CrossEntropyLoss()
        output = loss(predictions, labels)
        self.loss_list.append(output)
        return loss

    def accuracy(self, labels, predictions):
        correct_count = 0
        count = 0
        for i in range(len(predictions)):
            if np.argmax(i) == labels[i]:
                correct_count += 1
                count += 1
            else:
                count += 1
        
        return correct_count/count


    def train(self, inputs, labels):
        print("training")
        inputs = torch.tensor(inputs)
        print(inputs.shape)
        probabilites = self.call(inputs)
        print(probabilites.shape)

        loss = self.loss(labels, probabilites)
        loss.backward()


        
    def test(self):
        pass

def main():
    print("running")
    model = Model()
    train_inputs, test_inputs, train_labels, test_labels = preprocess()
    model.train(train_inputs, train_labels)

if __name__ == "__main__":
    main()
