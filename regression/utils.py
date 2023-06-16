import torch

# training and evaluation loop adapted from https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/4e865243430a47a00d551ca0579a6f6c/cifar10_tutorial.ipynb
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