import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import copy
import numpy as np
from torchvision import datasets, transforms, models
from torch.optim import lr_scheduler
from torch import nn
import json

# argparse module for train script
parser = argparse.ArgumentParser(description='parser for image calssifier. train.py script')
parser.add_argument('--data_dir', type=str, action='store', dest='data_dir', default='flowers')
parser.add_argument('--arch', type=str, action='store', dest='arch', default='vgg16', choices=['vgg16', 'vgg13'])
parser.add_argument('--learning_rate', type=float, action='store', dest='learning_rate', default=0.001)
parser.add_argument('--hidden_units', type=int, action='store', dest='hidden_units', default=512)
parser.add_argument('--epochs', type=int, action='store', dest='epochs', default=15)
parser.add_argument('--save_dir', type=str, action='store', dest='save_dir')
parser.add_argument('--gpu', type=str, default='gpu', choices=['gpu', 'cpu'])



args = parser.parse_args()

# check hidden_units input
if args.hidden_units % 2 != 0:
    parser.error("please use a number divisable by two for the number of hidden units")


# setting up subfolder of data directory
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'



# Defining the transforms for the training, validation, and testing sets
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ])}

# Load the datasets with ImageFolder
image_datasets = {x: datasets.ImageFolder(data_dir + '/' + x, 
                                          data_transforms[x]) for x in ['train', 'valid', 'test']}

# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                              batch_size=32,
                                              shuffle=True) for x in ['train', 'valid', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}



# label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    

# load pretrained model
if args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
if args.arch == 'vgg13':
    model = models.vgg13(pretrained=True)



# set up classifier specifications
hidden_units_one = int(args.hidden_units)
hidden_units_two = int(hidden_units_one / 2)


# set device from command line parser
if args.gpu == 'gpu':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

print('Set Device: ', device)

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(
    nn.Linear(25088, 1024),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(1024, hidden_units_one),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(hidden_units_one, hidden_units_two),
    nn.ReLU(),
    nn.Linear(hidden_units_two, 102),
    nn.LogSoftmax(dim=1))

model.classifier = classifier

criterion = nn.NLLLoss()

# get learning_rate from command line parser
learning_rate = args.learning_rate

optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.05)

model = model.to(device)


def train_model(model, scheduler, criterion, optimizer, epochs=15):
    """
    This Funtion trains a model classifier from a pretrained deep learning model. 
    
        Arguments:
        :param model: pytorch model object
        :param scheduler: pytorch scheduler object for decay on learning rate after x steps
        :param criterion: pytorch loss function to use for training
        :param optimizer: pytorch optimizing algorithm object to use for training
        :param epochs: int number of training epochs (iterations)
        
        :return: trained pytorch model object
    """
    print_every = len(dataloaders['train'])
    best_accuracy = 0.0
    running_loss=0.0
    test_loss=0.0
    accuracy=0.0
    
    for epoch in range(epochs):
        print('-' * 10)
        steps=0
        scheduler.step()
        for inputs, labels in dataloaders['train']:
            steps += 1
            
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
                
            if steps == print_every:
                test_loss = 0.0
                accuracy = 0.0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        test_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                epoch_accuracy = accuracy
                if epoch_accuracy > best_accuracy:
                      best_accuracy = epoch_accuracy
                      best_model = copy.deepcopy(model.state_dict())
                running_loss = 0
                model.train()
                      
    model.load_state_dict(best_model)               
    return model



def test_model(model):
    """
    This function tests a trained model on the test data.
    
        Arguments:
        :param model: trained pytorch model object
        
        :return: None
    """
    accuracy = 0.0
    
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    final_accuracy = accuracy / len(dataloaders['test'])
    print('-' * 10)
    print('Final Accuracy on Test Data: {:.4f}'.format(final_accuracy))
    print('-' * 10)

epochs = args.epochs

model = train_model(model, scheduler, criterion, optimizer, epochs)
test_model(model)

checkpoint = {
    'model': model,
    'state_dict': model.state_dict(),
    'optimizer': optimizer,
    'scheduler': scheduler,
    'class_to_idx': image_datasets['train'].class_to_idx,
    'epochs': epochs
}

save_dir = args.save_dir

# saving trained model
if args.save_dir:
    torch.save(checkpoint, save_dir + '/checkpoint.pth')
else:
    torch.save(checkpoint, 'checkpoint.pth')
    
print('Model Checkpoint Saved')