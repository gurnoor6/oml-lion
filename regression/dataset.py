import torch
import numpy as np

# function taken from https://github.com/epfml/OptML_course/blob/master/labs/ex05/template/helpers.py
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x,axis = 0)
    x = x - mean_x
    std_x = np.std(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x


# code for loading data taken from https://github.com/epfml/OptML_course/blob/master/labs/ex05/template/Lab%205%20-%20Stochastic%20Gradient%20Descent.ipynb
def get_train_test_loaders(train_batch_size=8, test_batch_size=8):
    data = np.loadtxt("./regression/Concrete_Data.csv",delimiter=",")

    A = data[:,:-1]
    b = data[:,-1]

    A, mean_A, std_A = standardize(A)
    A = torch.FloatTensor(A)
    b = torch.FloatTensor(b)

    generator = torch.Generator().manual_seed(42)
    dataset = []
    for x, y in zip(A, b):
        dataset.append((x, y))

    trainset, testset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=False, generator=generator)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, generator=generator)

    return trainloader, testloader