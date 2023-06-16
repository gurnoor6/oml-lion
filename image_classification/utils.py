import torch

torch.manual_seed(0)

# train, test code taken from https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/4e865243430a47a00d551ca0579a6f6c/cifar10_tutorial.ipynb
def train(model, criterion, optimizer, trainloader, device, num_epochs=2):
  for epoch in range(num_epochs):  # loop over the dataset multiple times

      running_loss = 0.0
      for i, data in enumerate(trainloader, 0):
          # get the inputs
          inputs, targets = data

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = model(inputs.to(device))
          loss = criterion(outputs, targets.to(device))
          loss.backward()
          optimizer.step()

          running_loss += loss.item()

  return model

def eval(model, device, testloader):
  model = model.to(device)

  correct = 0
  total = 0
  with torch.no_grad():
      for data in testloader:
          images, labels = data
          outputs = model(images.to(device))
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels.to(device)).sum().item()
  
  return 100 * correct / total
