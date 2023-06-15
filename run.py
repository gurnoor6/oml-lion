
import torch
import torch.optim as optim
from model import Net
import torch.nn as nn
from utils import train, eval
from dataset import get_train_test_loaders
from lion import Lion

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

def lr_runs():
    sgd_accuracy = []
    lion_accuracy = []
    for lr in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        print(f"lr = {lr}")
        trainloader, testloader = get_train_test_loaders()
        sgd_accuracy.append(sgd(lr, trainloader, testloader))
        lion_accuracy.append(lion(lr, trainloader, testloader))

        print(f"sgd_accuracy = {sgd_accuracy}")
        print(f"lion_accuracy = {lion_accuracy}")


def main():
    lr_runs()


if __name__ == '__main__':
    main()