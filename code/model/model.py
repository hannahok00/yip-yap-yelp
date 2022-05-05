from cProfile import label
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
from matplotlib import pyplot as plt

#not sure of the equivalent to earlystopping or modelcheckpoint

#linear is equivalent to dense layers
#GRU layer would represent the bidirectional layer you can set GRU bidirectional = true 
#for global max pooling can do output, _ = torch.max(input, 1)

#constants
GPU = False
MAX_WORDS = 50
#Depends on which classification (2, 3 or 5)
NUMBER_OF_CLASSES = 3
#Set large vocab size for embedding matrix
VOCAB_SIZE = 500000
EPOCHS = 50
BATCH_SIZE = 100

class Model(torch.nn.Module):
    def __init__(self, classification):
        super(Model, self).__init__()

        #Define batch size
        self.batch_size = 100
        self.linear_size_one = 300 
        self.linear_size_two = 100 
        
        #Define embedding matrix
        self.embedding = Embedding(VOCAB_SIZE, 128)
        
        #Define layers
        self.LSTM = LSTM(MAX_WORDS*128, self.linear_size_one)
        self.l1 = Linear(self.linear_size_one, self.linear_size_two)
        self.relu = ReLU()

        #Pass in the number of output classes
        self.l2 = Linear(self.linear_size_two, classification)

        #Sigmoid for binary, softmax for multi-class
        self.softmax = Softmax()
        self.sigm= Sigmoid()
    
        #Loss list for plotting the losses
        self.loss_list = []

    def call(self, reviews):
        #The shape of the self.embedding output will be [sentence_length, batch_size, embedding_dim]
        l1_out = self.embedding(reviews)
        l1_out = torch.reshape(l1_out, (self.batch_size, MAX_WORDS*128))

        #Pass inputs through LSTM
        l2_out, hidden_state = self.LSTM(l1_out)
        
        #Pass through dense layers
        l3_out = self.l1(l2_out)
       
        l4_out = self.relu(l3_out)
       
        l5_out = self.l2(l4_out) 

        #Use softmax to get probability distribution 
        final_out = self.softmax(l5_out)

        
        return final_out

    def loss(self, labels, predictions):

        #Convert labels to tensor
        labels = torch.tensor(labels)
        
        #Define loss 
        loss = torch.nn.CrossEntropyLoss(size_average=False)

        #Calculate the loss
        output = loss(predictions, labels)
        
        return output


    def accuracy(self, labels, predictions):
        
        correct_count = 0
        count = 0
        #Run through each input in batch
        for i in range(len(predictions)):
            #Helps to indicate which predictions are correct/shows what the model learned for that input
            print(predictions[i])
            print(labels[i])
            print(torch.argmax(predictions[i]).item())
            #Returns the indices of the maximum value thus if correctly predicted increments counter
            if torch.argmax(predictions[i]).item() == labels[i]:
                correct_count += 1
                count += 1

            #Increments total counter if incorrectly predicted
            else:
                count += 1
        
        #Return the correct predictions over total to give accuracy for that batch
        return correct_count/count


    def train(self, inputs, labels):
        #Define optimizer to use in backpropogation
        optimizer = optim.Adam(self.parameters(), lr=0.005)

        
        for i in range(int(len(inputs)/self.batch_size)):
            print("Training batch: ", i)

            #Get the next batch of inputs and labels
            input_batch = inputs[i*self.batch_size: i*self.batch_size + self.batch_size]
            labels_batch = labels[i*self.batch_size: i*self.batch_size + self.batch_size]
   
            #Convert inputs to tensor
            input_batch = torch.tensor(input_batch)
           
            #Run forward pass
            probabilites = self.call(input_batch)

            #Calculate the loss
            loss = self.loss(labels_batch, probabilites)
            print("Loss from batch: ", i, "i", loss)
            self.loss_list.append(loss.item())
            
            #Update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return 


    
    def test(self, inputs, labels):
        accuracy_list = []

        for i in range(int(len(inputs)/self.batch_size)):
            print("Testing batch: ", i)

            #Get next batch of inputs and labels
            input_batch = inputs[i*self.batch_size: i*self.batch_size + self.batch_size]
            labels_batch = labels[i*self.batch_size: i*self.batch_size + self.batch_size]

            #Convert inputs to tensor to run through model
            input_batch = torch.tensor(input_batch)

            #Get the probability distribution for the batch
            probabilites = self.call(input_batch)

            #Calculate the accuracy
            accuracy = self.accuracy(labels_batch, probabilites)
            print("batch accuracy: ", accuracy)
            
            #Append accuracy to list
            accuracy_list.append(accuracy)
        
        #Return the average accuracy across all batches
        return np.average(accuracy_list)




def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    
    """
    #losses = losses.numpy()
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  

def main():
    
    print("called main")

    #Instantiate the model
    #Change classification to be number of classes you want model to differentiate between
    model = Model(classification=2)

    #Get the train and test inputs and labels from preprocess
    #Preprocesses labels depending on what type of classification: binary/multi-class
    train_inputs, test_inputs, train_labels, test_labels = preprocess(classification=2)

    #Train the model
    model.train(train_inputs, train_labels)

    #Get the accuracy from testing
    accuracy = model.test(test_inputs, test_labels)

    visualize_loss(model.loss_list)
    print("accuracy", accuracy)
     
if __name__ == "__main__":
    main()
