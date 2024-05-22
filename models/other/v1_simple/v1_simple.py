import numpy as np
import pandas
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from datetime import datetime
import time

class Data(Dataset):
  def __init__(self, X_train, y_train):
    # need to convert float64 to float32 else will get the following error
    # RuntimeError: expected scalar type Double but found Float
    self.X = torch.from_numpy(X_train.astype(np.float32))

    # need to convert float64 to Long else will get the following error
    # RuntimeError: expected scalar type Long but found Float
    self.y = torch.from_numpy(y_train).type(torch.LongTensor)
    self.len = self.X.shape[0]
  
  def __getitem__(self, index):
    return self.X[index], self.y[index]
  
  def __len__(self):
    return self.len

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.stack = nn.Sequential(
        nn.Linear(9, 64),
        nn.Sigmoid(),
        nn.Linear(64, 32),
        nn.Sigmoid(),
        nn.Linear(32, 16),
        nn.Sigmoid(),
        nn.Linear(16, 2)
        )

    def forward(self, x):
        x = self.stack(x)
        return x

if __name__ == '__main__':
    script_start = time.time()

    # Load the data
    brand = 3
    df = pandas.read_csv('data/brand%d.csv' % brand)

    # Process the data
    df = df.dropna()
    df = df.iloc[-100000:]
    df = df.drop(columns=['car'], axis=1)

    # get count of 0s and 1s in the dataset
    unique, counts = np.unique(df['label'], return_counts=True)
    print("Number of 0s:", counts[0])
    print("Number of 1s:", counts[1])
    print("Data proportion of 0s:", counts[0] / len(df), "\n")

    # Normalize all columns
    for column in df.columns:
        if column != 'label':
            df[column] = (df[column] - df[column].mean()) / df[column].std()

    # Seperate input and output data
    X = np.array(df.drop(['label'], axis=1))
    Y = np.array(df['label'])
    print("Processed data X:", X.shape)
    print("Processed data Y:", Y.shape, "\n")

    # Split the data into training, validation, and testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
    print("Training X:", X_train.shape)
    print("Training Y:", Y_train.shape)
    print("Validation X:", X_val.shape)
    print("Validation Y:", Y_val.shape)
    print("Testing X:", X_test.shape)
    print("Testing Y:", Y_test.shape, "\n")

    # Load the data into the DataLoader
    traindata = Data(X_train, Y_train)
    valdata = Data(X_val, Y_val)
    print("traindata size:", len(traindata))
    print("valdata size:", len(valdata))

    batch_size = 4096
    num_workers = 4

    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=num_workers, multiprocessing_context="forkserver", persistent_workers=True)
    valloader = DataLoader(valdata, batch_size=batch_size, shuffle=True, num_workers=num_workers, multiprocessing_context="forkserver", persistent_workers=True)
    print("trainloader size:", len(trainloader))
    print("valloader size:", len(valloader))

    # Create the model
    clf = Network()
    print(clf.parameters)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(clf.parameters(), lr=0.1, momentum=0.9)

    epochs = 256
    train_accuracy = []
    val_accuracy = []

    for epoch in range(epochs):
        epcoh_start = time.time()
        running_loss = 0.0
        train_correct, train_total = 0, 0

        for data in trainloader:
            inputs, labels = data

            # set optimizer to zero grad to remove previous epoch gradients
            optimizer.zero_grad()

            # forward propagation
            outputs = clf(inputs)
            loss = criterion(outputs, labels)

            # backward propagation
            loss.backward()

            # optimize
            optimizer.step()
            running_loss += loss.item()

            # get training accuracy
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy.append(train_correct / train_total)

        # get validation accuracy
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for data in valloader:

                inputs, labels = data

                outputs = clf(inputs)
                
                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy.append(val_correct / val_total)

        # print statistics
        print(f"Epoch: {epoch + 1}, Training Accuracy: {np.round(train_accuracy[-1], 4)}, Validation Accuracy: {np.round(val_accuracy[-1], 4)}")
        print(f"Time taken: {time.time() - epcoh_start}")

    # save plot of training and validation accuracy over epochs
    model_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.plot(train_accuracy[1:], label='Training Accuracy')
    plt.plot(val_accuracy[1:], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig('./models/v1_simple/results/' + model_timestamp + '_brand' + str(brand) + '.png')

    # save model with timestamp in file name
    model_path = './models/v1_simple/results/' + model_timestamp + '_brand' + str(brand) + '.pth'
    torch.save(clf.state_dict(), model_path)

    # load model
    clf = Network()
    clf.load_state_dict(torch.load(model_path))

    # load test data
    testdata = Data(X_test, Y_test)
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True, num_workers=num_workers, multiprocessing_context="forkserver", persistent_workers=True)

    # get test dataset proportion of 0s and 1s
    unique, counts = np.unique(Y_test, return_counts=True)
    print("Test data proportion of 0s:", counts[0] / len(Y_test))
    print("Test data proportion of 1s:", counts[1] / len(Y_test))

    # evaluate the model on the test data
    correct, total = 0, 0

    # no need to calculate gradients during inference
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data

            # calculate output by running through the network
            outputs = clf(inputs)

            # get the predictions
            __, predicted = torch.max(outputs.data, 1)

            # update results
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {len(testdata)} test data: {100 * correct / total} %')

    script_end = time.time()
    print(f"Total time taken: {script_end - script_start} seconds")