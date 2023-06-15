
import torch
import numpy as np
import torch.optim as optim
from model import Net
import torch.nn as nn
from utils import train, eval
from dataset import get_train_test_loaders
from lion import Lion
from sophia import SophiaG

torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # We will run on CUDA if there is a GPU available
print("run: ", device)

def sgd(lr, trainloader, testloader):
    print("SGD")
    model = Net() # init
    model.to(device)
    print(f"model has {torch.nn.utils.parameters_to_vector(model.parameters()).numel()} learnable parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    model = train(model, criterion, optimizer, trainloader, device)
    accuracy = eval(model, device, testloader)

    return accuracy

def lion(lr, trainloader, testloader):
    print("LION")
    model = Net() # init
    model.to(device)
    print(f"model has {torch.nn.utils.parameters_to_vector(model.parameters()).numel()} learnable parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = Lion(model.parameters(), lr=lr)

    model = train(model, criterion, optimizer, trainloader, device)
    accuracy = eval(model, device, testloader)

    return accuracy

def sophia(lr, trainloader, testloader):
    print("SOPHIA")
    model = Net() # init
    model.to(device)
    print(f"model has {torch.nn.utils.parameters_to_vector(model.parameters()).numel()} learnable parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = SophiaG(model.parameters(), lr=lr)

    model = train(model, criterion, optimizer, trainloader, device)
    accuracy = eval(model, device, testloader)

    return accuracy

def lr_runs():
    # perform the experiment 10 times
    sgd_accuracy_mean = []
    lion_accuracy_mean = []
    sophia_accuracy_mean = []
    for _ in range(10):
        sgd_accuracy = []
        lion_accuracy = []
        sophia_accuracy = []
        for lr in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            print(f"lr = {lr}")
            trainloader, testloader = get_train_test_loaders()
            sgd_accuracy.append(sgd(lr, trainloader, testloader))
            lion_accuracy.append(lion(lr, trainloader, testloader))
            sophia_accuracy.append(sophia(lr, trainloader, testloader))

            print(f"sgd_accuracy = {sgd_accuracy}")
            print(f"lion_accuracy = {lion_accuracy}")
            print(f"sophia_accuracy = {sophia_accuracy}")
        
        sgd_accuracy_mean.append(np.array(sgd_accuracy))
        lion_accuracy_mean.append(np.array(lion_accuracy))
        sophia_accuracy_mean.append(np.array(sophia_accuracy))

    print("SGD accuracy mean: ", np.mean(sgd_accuracy_mean, axis=0))
    print("LION accuracy mean: ", np.mean(lion_accuracy_mean, axis=0))
    print("SOPHIA accuracy mean: ", np.mean(sophia_accuracy_mean, axis=0))    

def batch_size_runs():
    sgd_accuracy = []
    lion_accuracy = []
    sophia_accuracy = []
    for batch_size in [32, 64, 128, 256, 512, 1024]:
        print(f"batch_size = {batch_size}")
        trainloader, testloader = get_train_test_loaders(train_batch_size=batch_size)
        sgd_accuracy.append(sgd(1e-4, trainloader, testloader))
        lion_accuracy.append(lion(1e-4, trainloader, testloader))
        sophia_accuracy.append(sophia(1e-4, trainloader, testloader))

        print(f"sgd_accuracy = {sgd_accuracy}")
        print(f"lion_accuracy = {lion_accuracy}")
        print(f"sophia_accuracy = {sophia_accuracy}")

def main():
    lr_runs()


if __name__ == '__main__':
    main()