# Imports here
import torch
import json
from torch import nn
import numpy as np
from torch import optim
from torchvision import datasets, models, transforms
from torch import device, utils
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import argparse

parser = argparse.ArgumentParser("Arguments for the train.py file")

parser.add_argument("train", type=str, help="The data you want your model to train on")
parser.add_argument("validation", type=str, help="The data you want your model to do the validation on")
parser.add_argument("testing", type=str, help="The data you want your model to do the testing on")
parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
parser.add_argument("--model", type=str, default="densenet121", help="pretrained model")
parser.add_argument("--fc1", type=int, default=563, help="hidden layers")
parser.add_argument("--fc2", type=int, default=102, help="hidden layers")


# maybe model architecture to edit
# parser.add_argument("--model", type=str, default="densenet121", help="pretrained model")

# learning rate for optimizer
parser.add_argument("--lr", type=int, default=0.003, help="optimizer learning rate")
# epochs for training
parser.add_argument("--epochs", type=int, default=3, help="choose number of epochs")

# checkpoint path
parser.add_argument("checkpoint", type=str, help="choose where the checkpoint will be stored")

args = parser.parse_args()


# Loading the data
""" don' t forget to modify this because paths are going to be different
"""
#data_dir = 'flowers'
#train_dir = data_dir + '/train'
#valid_dir = data_dir + '/valid'
#test_dir = data_dir + '/test'
data = [args.train, args.validation, args.testing]

# TODO: Define your transforms for the training, validation, and testing sets
trainning_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])
                                     ])
test_transforms = transforms.Compose([transforms.Resize(254),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])
                                     ])



# TODO: Load the datasets with ImageFolder
trainning_datasets = datasets.ImageFolder(data[0], transform=trainning_transforms)# edit
valid_datasets = datasets.ImageFolder(data[1], transform=test_transforms)# edit
test_datasets = datasets.ImageFolder(data[2], transform=test_transforms)# edit

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(trainning_datasets, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=64)
testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64)

# Label mapping

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Building and training the classifier

# TODO: Build and train your network$
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(args.device) # edited
# model = models.densenet121(pretrained=True)
model = eval("models.{}(pretrained=True)".format(args.model)) # edited

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(1024, args.fc1),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(563, args.fc2),
                                 nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
# optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)  # edited
model.to(device);

# epochs = 3
epochs = args.epochs  # edited
trainning_loss = 0
print_every = 5
steps = 0

for epoch in range(epochs):
    for images, labels in trainloader:
        steps += 1
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        mbps = model(images)
        loss = criterion(mbps, labels)
        loss.backward()
        optimizer.step()

        trainning_loss += loss.item()

        if steps % print_every == 0: # maybe edit
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)
                    logps = model(images)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            print(f"Epoch {epoch + 1}/{epochs}.. "
                  f"Train loss: {trainning_loss / print_every:.3f}.. "
                  f"Test loss: {test_loss / len(validloader):.3f}.. "
                  f"Test accuracy: {accuracy / len(validloader):.3f}")
            trainning_loss = 0
            model.train()

# testing the network

# TODO: Do validation on the test set
test_loss = 0
accuracy = 0
model.eval()
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        logps = model(images)
        loss = criterion(logps, labels)
        test_loss += loss.item()

        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print("test_loss: {}, accuracy: {}".format(test_loss / len(testloader), accuracy / len(testloader)))
        test_loss = 0

# save the checkpoint
# edit
# TODO: Save the checkpoint
model.class_to_idx = trainning_datasets.class_to_idx
checkpoint = {"input_size": 1024,
             "output_size": 102,
             "hidden_layers": [layer.out_features for layer in model.classifier if layer.__class__.__name__ == "Layer"],
             "epoch": epochs,
             "optimizer_state_dict": optimizer.state_dict(),
             "model": model,
             "class_to_idx":model.class_to_idx}

torch.save(checkpoint, "checkpoint.pth")

