import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import SGD, Adam
from regression.model import Net
from optimizers.lion import Lion
from optimizers.sophia import SophiaG
from regression.utils import train, eval
from regression.dataset import get_train_test_loaders
from plots.plot import plot_lr_accuracy, plot_bs_accuracy

torch.manual_seed(42)

def sgd(lr, trainloader, testloader):
    """
    Method to train the model using SGD
    """
    print("SGD")
    model = Net() # init

    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)

    model = train(model, criterion, optimizer, trainloader)
    loss = eval(model, criterion, testloader)

    return loss

def lion(lr, trainloader, testloader):
    """
    Method to train the model using Lion
    """
    print("LION")
    model = Net() # init

    criterion = nn.MSELoss()
    optimizer = Lion(model.parameters(), lr=lr)

    model = train(model, criterion, optimizer, trainloader)
    loss = eval(model, criterion, testloader)

    return loss

def sophia(lr, trainloader, testloader):
    """
    Method to train the model using Sophia
    """
    print("SOPHIA")
    model = Net() # init

    criterion = nn.MSELoss()
    optimizer = SophiaG(model.parameters(), lr=lr)

    model = train(model, criterion, optimizer, trainloader)
    loss = eval(model, criterion, testloader)

    return loss

def adam(lr, trainloader, testloader):
    """
    Method to train the model using Adam
    """
    print("ADAM")
    model = Net() # init

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    model = train(model, criterion, optimizer, trainloader)
    loss = eval(model, criterion, testloader)

    return loss


def lr_runs(lrs=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]):
    trainloader, testloader = get_train_test_loaders()

    sgd_loss = []
    lion_loss = []
    sophia_loss = []
    adam_loss = []
    for lr in tqdm(lrs):
        sgd_loss.append(sgd(lr, trainloader, testloader))
        lion_loss.append(lion(lr, trainloader, testloader))
        sophia_loss.append(sophia(lr, trainloader, testloader))
        adam_loss.append(adam(lr, trainloader, testloader))
    
    print("SGD: ", sgd_loss)
    print("Lion: ", lion_loss)
    print("Sophia: ", sophia_loss)
    print("Adam: ", adam_loss)

    plot_lr_accuracy(sgd_loss, lion_loss, sophia_loss, adam_loss, lrs, "Loss", "regressionlr.png")

def batch_size_runs(batch_sizes=[8, 16, 32, 64, 128, 256]):
    sgd_loss = []
    lion_loss = []
    sophia_loss = []
    adam_loss = []

    for batch_size in tqdm(batch_sizes):
        trainloader, testloader = get_train_test_loaders(train_batch_size=batch_size)
        sgd_loss.append(sgd(0.01, trainloader, testloader))
        lion_loss.append(lion(0.01, trainloader, testloader))
        sophia_loss.append(sophia(0.01, trainloader, testloader))
        adam_loss.append(adam(0.01, trainloader, testloader))
    
    print("SGD: ", sgd_loss)
    print("Lion: ", lion_loss)
    print("Sophia: ", sophia_loss)
    print("Adam: ", adam_loss)

    plot_bs_accuracy(sgd_loss, lion_loss, sophia_loss, adam_loss, batch_sizes, "Loss", "regressionbs.png")

if __name__ == "__main__":
    lr_runs()
    batch_size_runs()