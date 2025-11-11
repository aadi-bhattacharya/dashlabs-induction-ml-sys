import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class mModel(nn.Module):
    def __init__(s):
        nn.Module.__init__(s)
        s.layer1 = nn.Linear(28*28, 64)
        s.layer2 = nn.Linear(64, 10)

    def forward(s, x):
        x = x.view(-1, 28*28)
        x = nn.functional.relu(s.layer1(x))
        x = s.layer2(x)
        return nn.functional.log_softmax(x, dim=1)

def trainingc(model, data_loader, lr=0.1, epochs=2):
    opt = optim.SGD(model.parameters(), lr)
    model.train()
    for e in range(epochs):
        for imgs, labels in data_loader:
            opt.zero_grad()
            out = model(imgs)
            loss = nn.functional.nll_loss(out, labels)
            loss.backward()
            opt.step()
    return model.state_dict()

def average(models):
    avg = {}
    for i in models[0]:
        s = 0
        for m in models:
            s +=m[i]
        avg[i] = s / len(models)
    return avg

def check_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            out = model(imgs)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    print("Accuracy:", 100 * correct / total, "%")

def federated():
    trans = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST('.', train=True, download=True, transform=trans)
    test = datasets.MNIST('.', train=False, transform=trans)
    loader = DataLoader(test, batch_size=64, shuffle=False)

    nclients = 2
    size = len(train) // nclients
    parts = random_split(train, [size]*nclients)
    loaders = [DataLoader(p, batch_size=32, shuffle=True) for p in parts]

    globalm = mModel()

    for itr in range(2):
        print("iteration", itr+1)
        new_weights = []
        for i in range(nclients):
            print("client",i+1,"training")
            localm = mModel()
            localm.load_state_dict(globalm.state_dict())
            trained = trainingc(localm, loaders[i])
            new_weights.append(trained)
        globalm.load_state_dict(average(new_weights))
    check_accuracy(globalm, loader)

federated()
