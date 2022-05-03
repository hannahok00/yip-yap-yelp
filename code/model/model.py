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
#Depends on binary classification or not
NUMBER_OF_CLASSES = 3
#Idk what vocab size is 
VOCAB_SIZE = 500000
EPOCHS = 50
BATCH_SIZE = 1024

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        #Define batch size
        self.batch_size = 100
        
        #Need to figure out vocab size, define embedding matrix
        self.embedding = Embedding(VOCAB_SIZE, 128)
        
        #Define layers
        self.LSTM = LSTM(MAX_WORDS*128, 300)
        self.l1 = Linear(300, 100)
        self.relu = ReLU()
        self.l2 = Linear(100, NUMBER_OF_CLASSES)

        #Do we want softmax or sigmoid
        self.softmax = Softmax()
        self.sigm= Sigmoid()
    

        self.loss_list = []

    def call(self, reviews):
        
        
        #The shape of the self.embedding output will be [sentence_length, batch_size, embedding_dim]
        l1_out = self.embedding(reviews)
        print("l1 output shape:", l1_out.shape)

        #Reshape output to be (100, 2560) which is dimension (sentence_length * embedding_dim)
        #?? This component is questionable
        l1_out = torch.reshape(l1_out, (self.batch_size, MAX_WORDS*128))
        print("l1 output shape:", l1_out.shape)

        #Pass inputs through LSTM
        l2_out, hidden_state = self.LSTM(l1_out)
        print("l2 output shape:", l2_out.shape)

        #Pass through dense layers
        l3_out = self.l1(l2_out)
        print("l3 output shape:", l3_out.shape)
        l4_out = self.relu(l3_out)
        print("l4 output shape:", l4_out.shape)
        l5_out = self.l2(l4_out) 
        print("l5 output shape:", l5_out.shape)

        #Use sigmoid to get probabilities if binary, softmax if ternary 
        final_out = self.softmax(l5_out)
        print("final output shape:", final_out.shape)
        
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
            print(predictions[i])
            print(labels[i])
           # print(torch.argmax(predictions[i]))
            #Returns the indices of the maximum value thus if correctly predicted increments counter
            if torch.argmax(predictions[i]) == labels[i]:
                correct_count += 1
                count += 1
            #Increments total counter if incorrectly predicted
            else:
                count += 1
        
        #Return the correct predictions over total to give accuracy
        return correct_count/count


    def train(self, inputs, labels):
        #Define optimizer to use in backpropogation
        optimizer = optim.Adam(self.parameters(), lr=0.005)

        
        for i in range(int(len(inputs)/self.batch_size)):
            print("Training batch: ", i)

            #Get the next batch of inputs and labels
            input_batch = inputs[i*self.batch_size: i*self.batch_size + self.batch_size]
            labels_batch = labels[i*self.batch_size: i*self.batch_size + self.batch_size]
            #labels_batch = [label - 1 for label in labels_batch]
            #Convert inputs to batch
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
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
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
    model = Model()

    #Get the train and test inputs and labels from preprocess
    train_inputs, test_inputs, train_labels, test_labels = preprocess(multi_class=False)

    #Train the model
    model.train(train_inputs, train_labels)

    #Get the accuracy from testing
    accuracy = model.test(test_inputs, test_labels)

    visualize_loss(model.loss_list)
    print("accuracy", accuracy)
     
if __name__ == "__main__":
    main()
