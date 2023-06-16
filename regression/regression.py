import torch
import numpy as np
import torch.nn as nn
torch.manual_seed(42)
import matplotlib.pyplot as plt
from torch.optim import SGD, Adam
from lion import Lion
from sophia import SophiaG


# function taken from https://github.com/epfml/OptML_course/blob/master/labs/ex05/template/helpers.py
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x,axis = 0)
    x = x - mean_x
    std_x = np.std(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x


# code for loading data taken from https://github.com/epfml/OptML_course/blob/master/labs/ex05/template/Lab%205%20-%20Stochastic%20Gradient%20Descent.ipynb
data = np.loadtxt("./Concrete_Data.csv",delimiter=",")

A = data[:,:-1]
b = data[:,-1]
A, mean_A, std_A = standardize(A)
A = torch.FloatTensor(A)
b = torch.FloatTensor(b)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 1)

    def forward(self, x):
        y = self.linear(x)
        return y
    
generator = torch.Generator().manual_seed(42)
dataset = []
for x, y in zip(A, b):
  dataset.append((x, y))
trainset, testset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=False, generator=generator)
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, generator=generator)

def train(model, criterion, optimizer, trainloader):
  for epoch in range(20):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        outputs = outputs.reshape(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

  print('Finished Training')
  return model

def eval(model, criterion, testloader):
  running_loss = 0.0
  with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # forward + backward + optimize
        outputs = model(inputs)
        outputs = outputs.reshape(-1)
        loss = criterion(outputs, labels)

        # print statistics
        running_loss += loss.item()
  
  return running_loss / len(testloader)

criterion = nn.MSELoss()

def lr_run():
    sgd_loss = []
    lion_loss = []
    sophia_loss = []
    adam_loss = []
    for lr in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        model = Net()
        optimizer = SGD(model.parameters(), lr=lr)
        model = train(model, criterion, optimizer, trainloader)
        loss = eval(model, criterion, testloader)
        sgd_loss.append(loss)

        model = Net()
        optimizer = Lion(model.parameters(), lr=lr)
        model = train(model, criterion, optimizer, trainloader)
        loss = eval(model, criterion, testloader)
        lion_loss.append(loss)

        model = Net()
        optimizer = SophiaG(model.parameters(), lr=lr)
        model = train(model, criterion, optimizer, trainloader)
        loss = eval(model, criterion, testloader)
        sophia_loss.append(loss)

        model = Net()
        optimizer = Adam(model.parameters(), lr=lr)
        model = train(model, criterion, optimizer, trainloader)
        loss = eval(model, criterion, testloader)
        adam_loss.append(loss)


    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    lrs = list(map(lambda x: str(x), [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]))
    plt.plot(lrs, sgd_loss, label="SGD")
    plt.plot(lrs, lion_loss, label="Lion")
    plt.plot(lrs, sophia_loss, label="Sophia")
    plt.plot(lrs, adam_loss, label="Adam")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig("regressionlr.png", dpi=500)

def batch_size_run():
    sgd_loss = []
    lion_loss = []
    sophia_loss = []
    adam_loss = []

    for batch_size in [8, 16, 32, 64, 128, 256]:
        model = Net()
        optimizer = SGD(model.parameters(), lr=1e-1)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, generator=generator)
        model = train(model, criterion, optimizer, trainloader)
        loss = eval(model, criterion, testloader)
        sgd_loss.append(loss)

        model = Net()
        optimizer = Lion(model.parameters(), lr=1e-1)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, generator=generator)
        model = train(model, criterion, optimizer, trainloader)
        loss = eval(model, criterion, testloader)
        lion_loss.append(loss)

        model = Net()
        optimizer = SophiaG(model.parameters(), lr=1e-1)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, generator=generator)
        model = train(model, criterion, optimizer, trainloader)
        loss = eval(model, criterion, testloader)
        sophia_loss.append(loss)

        model = Net()
        optimizer = Adam(model.parameters(), lr=1e-1)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, generator=generator)
        model = train(model, criterion, optimizer, trainloader)
        loss = eval(model, criterion, testloader)
        adam_loss.append(loss)
    
    plt.figure()
    plt.xlabel("Batch size")
    plt.ylabel("Loss")
    batch_sizes = list(map(lambda x: str(x), [8, 16, 32, 64, 128, 256]))
    plt.plot(batch_sizes, sgd_loss, label="SGD")
    plt.plot(batch_sizes, lion_loss, label="Lion")
    plt.plot(batch_sizes, sophia_loss, label="Sophia")
    plt.plot(batch_sizes, adam_loss, label="Adam")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.savefig("regressionbs.png", dpi=500)

if __name__ == "__main__":
    lr_run()
    batch_size_run()