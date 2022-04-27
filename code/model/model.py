import pickle
from pickletools import optimize
from random import shuffle
from xml.etree.ElementPath import prepare_parent
import numpy as np
import pandas as pd
import torch
from torch import max
from torch import optim
from torch.nn import LSTM, Linear, Dropout, MaxPool1d, GRU, Conv1d, Embedding, Sequential, ReLU, Softmax, Sigmoid
from real_preprocess import preprocess
from torch.utils.data import DataLoader

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

        self.batch_size = 100
        #self.model = Sequential()
        #Need to figure out vocab size
        self.embedding = Embedding(VOCAB_SIZE, 128)
        #self.l_test=Linear(128,1)
        #self.sigm= Sigmoid()
        self.LSTM = LSTM(2560, 300)
        self.l1 = Linear(300, 100)
        self.relu = ReLU()
        self.l2 = Linear(100, 2)
        self.softmax = Softmax()
    
        #self.sigm = Sigmoid()
        self.optimizer = optim.Adam(self.parameters(), lr=0.005)

        self.loss = torch.nn.CrossEntropyLoss()

        self.loss_list = []

    def call(self, reviews):
        
        #Need to figure out vocab size
        #The shape of the self.embedding will be [sentence_length, batch_size, embedding_dim]
        l1_out = self.embedding(reviews)
        print("l1 output shape:", l1_out.shape)
        l1_out = torch.reshape(l1_out, (self.batch_size, 20*128))
        print("l1 output shape:", l1_out.shape)
        l2_out, hidden_state = self.LSTM(l1_out)
        print("l2 output shape:", l2_out.shape)
        l3_out = self.l1(l2_out)
        print("l3 output shape:", l3_out.shape)
        l4_out = self.relu(l3_out)
        print("l4 output shape:", l4_out.shape)
        l5_out = self.l2(l4_out) 
        print("l5 output shape:", l5_out.shape)
        final_out = self.softmax(l5_out)
        print("final output shape:", final_out.shape)
        #print(final_out)
        return final_out

    def loss(self, labels, predictions):
        #Might need to reshape datat
        labels = torch.tensor(labels)
        print("labels shape", labels.shape)
        print("predictions shape", predictions.shape)
        loss = torch.nn.CrossEntropyLoss(size_average=False)
        output = loss(predictions, labels)
        self.loss_list.append(output)
        return output

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

        optimizer = optim.Adam(self.parameters(), lr=0.005)

        for i in range(int(len(inputs)/self.batch_size)):
            print(i)
            input_batch = inputs[i*self.batch_size: i*self.batch_size + self.batch_size]
            labels_batch = labels[i*self.batch_size: i*self.batch_size + self.batch_size]
            print(input_batch.shape)
            print(labels_batch.shape)
            print("training")

            input_batch = torch.tensor(input_batch)
           
            probabilites = self.call(input_batch)

            print(probabilites.shape)

            loss = self.loss(labels_batch, probabilites)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return 


        
    def test(self):
        pass

def main():
    print("running")
    model = Model()
    train_inputs, test_inputs, train_labels, test_labels = preprocess()
    
    #train__inputs_loader = DataLoader(train_inputs, batch_size=100, shuffle=False)
    #train_labels_loader = DataLoader(train_labels, batch_size=100, shuffle=False)
    
    model.train(train_inputs, train_labels)

if __name__ == "__main__":
    main()
