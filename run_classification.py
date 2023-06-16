
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from image_classification.model import Net
from optimizers.lion import Lion
from optimizers.sophia import SophiaG
from image_classification.utils import train, eval
from plots.plot import plot_lr_accuracy, plot_bs_accuracy
from image_classification.dataset import get_train_test_loaders

torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("run: ", device)

def sgd(lr, trainloader, testloader):
    """
    Method to train the model using SGD
    """
    print("SGD")
    model = Net() # init
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    model = train(model, criterion, optimizer, trainloader, device)
    accuracy = eval(model, device, testloader)

    return accuracy

def lion(lr, trainloader, testloader):
    """
    Method to train the model using Lion
    """
    print("LION")
    model = Net() # init
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Lion(model.parameters(), lr=lr)

    model = train(model, criterion, optimizer, trainloader, device)
    accuracy = eval(model, device, testloader)

    return accuracy

def sophia(lr, trainloader, testloader):
    """
    Method to train the model using Sophia
    """
    print("SOPHIA")
    model = Net() # init
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SophiaG(model.parameters(), lr=lr)

    model = train(model, criterion, optimizer, trainloader, device)
    accuracy = eval(model, device, testloader)

    return accuracy

def adam(lr, trainloader, testloader):
    """
    Method to train the model using Adam
    """
    print("ADAM")
    model = Net() # init
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model = train(model, criterion, optimizer, trainloader, device)
    accuracy = eval(model, device, testloader)

    return accuracy

def lr_runs(lrs=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]):
    """
    Method to run the experiment for different learning rates
    for all optimizers
    """
    print("lr_runs")
    sgd_accuracy = []
    lion_accuracy = []
    sophia_accuracy = []
    adam_accuracy = []

    for lr in tqdm(lrs):
        print(f"lr = {lr}")
        trainloader, testloader = get_train_test_loaders()
        sgd_accuracy.append(sgd(lr, trainloader, testloader))
        lion_accuracy.append(lion(lr, trainloader, testloader))
        sophia_accuracy.append(sophia(lr, trainloader, testloader))
        adam_accuracy.append(adam(lr, trainloader, testloader))
        

    print(f"sgd_accuracy = {sgd_accuracy}")
    print(f"lion_accuracy = {lion_accuracy}")
    print(f"sophia_accuracy = {sophia_accuracy}")
    print(f"adam_accuracy = {adam_accuracy}")
    plot_lr_accuracy(sgd_accuracy, lion_accuracy, sophia_accuracy, adam_accuracy, lrs, "Accuracy (in %)", "classlr.png")
    

def batch_size_runs(batch_sizes=[32, 64, 128, 256, 512, 1024]):
    """
    Method to run the experiment for different batch sizes
    for all optimizers
    """
    print("batch_size_runs")
    sgd_accuracy = []
    lion_accuracy = []
    sophia_accuracy = []
    adam_accuracy = []

    for bs in tqdm(batch_sizes):
        print(f"bs = {bs}")
        trainloader, testloader = get_train_test_loaders(train_batch_size=bs, test_batch_size=bs)
        sgd_accuracy.append(sgd(1e-1, trainloader, testloader))
        lion_accuracy.append(lion(1e-3, trainloader, testloader))
        sophia_accuracy.append(sophia(1e-3, trainloader, testloader))
        adam_accuracy.append(adam(1e-3, trainloader, testloader))
        

    print(f"sgd_accuracy = {sgd_accuracy}")
    print(f"lion_accuracy = {lion_accuracy}")
    print(f"sophia_accuracy = {sophia_accuracy}")
    print(f"adam_accuracy = {adam_accuracy}")
    plot_bs_accuracy(sgd_accuracy, lion_accuracy, sophia_accuracy, adam_accuracy, batch_sizes, "Accuracy (in %)", "classbs.png")


def main():
    lr_runs()
    batch_size_runs()


if __name__ == '__main__':
    main()