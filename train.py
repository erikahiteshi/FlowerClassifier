# Imports here
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image
import json
from matplotlib.ticker import FormatStrFormatter
import json
import argparse

#global
use_gpu = torch.cuda.is_available()

def load_data(args):
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(30),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(225),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(225),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return trainloader, validloader, testloader, train_data

def open_categoryname():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def create_model(args, cat_to_name, train_data):
    model = models.densenet121(pretrained=True)
    hidden_layers = args.hidden_layers
    learning_rate = args.lr
    for param in model.parameters():
        param.requires_grad = False
    classifier_input_size = model.classifier.in_features
    output_size = len(cat_to_name)
    model.classifier = nn.Sequential(nn.Linear(classifier_input_size, hidden_layers),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_layers, output_size),
                                 nn.LogSoftmax(dim=1))

    model.class_to_idx = train_data.class_to_idx
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    return model, hidden_layers, learning_rate, criterion, optimizer

def train_model(args, model, trainloader, testloader, validloader, criterion, optimizer):
    data_iter = iter(testloader)
    images, labels = next(data_iter)
    if use_gpu and args.gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    epochs = args.epochs
    steps = 0
    print_every = 50
    running_loss = 0

    print(f'Using the {device} to train')
    print("Training process initializing .....\n")

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            #now look at our validation data
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                training_loss = running_loss/print_every
                validation_loss = valid_loss/len(validloader)
                accuracy_loss = accuracy/len(validloader)
                print(f"Training on epoch {epoch+1}/{epochs}  |  "
                      f"Train Loss: {running_loss/print_every:.3f}  |  "
                      f"Validation Loss: {valid_loss/len(validloader):.3f}  |  "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    return training_loss, validation_loss, accuracy_loss
    
def save_checkpoint(args, model, train_data, optimizer):
    
    model.class_to_idx = train_data.class_to_idx
    checkpoint_path = 'densenet121_checkpoint.pth'
    
    checkpoint = {'arch': args.arch,
                 'input_size': 25088,
                 'classifier': model.classifier,
                 'output_size': 102,
                 'learning_rate': args.lr,
                 'class_to_idx': model.class_to_idx,
                 'hidden_layers': args.hidden_layers,
                 'epochs': args.epochs,
                 'optimizer': optimizer.state_dict(),
                 'state_dict': model.state_dict()}


    torch.save(checkpoint, checkpoint_path)
    print('Checkpoint saved')

def train_wrapper(args):
    trainloader, validloader, testloader, train_data = load_data(args)
    images, labels = next(iter(trainloader))
    cat_to_name = open_categoryname()
    model, hidden_layers, learning_rate, criterion, optimizer = create_model(args, cat_to_name, train_data)
    training_loss, validation_loss, accuracy_loss = train_model(args, model, trainloader, testloader, validloader, criterion, optimizer)
    save_checkpoint(args, model, train_data, optimizer)
    
def main():
    parser = argparse.ArgumentParser(description='Flower Classification trainer')
    parser.add_argument('--gpu', type=bool, default=False, help='Is GPU available')
    parser.add_argument('--arch', type=str, default='densenet', help='architecture [available: densenet, vgg]', required=True)
    parser.add_argument('--hidden_layers', type=int, default=512, help='Number of hidden layers in model')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
    parser.add_argument('--saved_model' , type=str, default='densenet121_checkpoint.pth', help='path of your saved model')
    args = parser.parse_args()
    train_wrapper(args)
    


if __name__ == '__main__':
    main()