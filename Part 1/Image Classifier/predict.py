import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.optim import lr_scheduler
from torch import nn
from PIL import Image
import json


parser = argparse.ArgumentParser(description='parser for image calssifier. train.py script')
parser.add_argument('--input', type=str, dest='image')
parser.add_argument('--model', type=str, dest='model')
parser.add_argument('--categrory_names', type=str, dest='categories', default='cat_to_name.json')
parser.add_argument('--gpu', type=str, default='gpu', choices=['gpu', 'cpu'])
parser.add_argument('--top_k', type=int, dest='top_k', default=1)
args = parser.parse_args()


# define variables from command line parser
model_path = args.model


def load_checkpoint(filename='checkpoint.pth'):
    """
    This function load a saved pytorch model to train and use
    
        Arguments:
        :param filename: str, filepath of .pth file to use
        
        return:
            load_model: loaded trained pytorch model
            load_ompimizer: loaded pytorch optimizer
            load_epoch: loaded number of epochs
            load_scheduler: loaded scheduler
    """
    print("=> loading checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename)
    load_epoch = checkpoint['epochs']
    
    load_model = checkpoint['model']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    load_model = load_model.to(device)
    load_model.class_to_idx = checkpoint['class_to_idx']
    load_model.load_state_dict(checkpoint['state_dict'])
    
    load_optimizer = checkpoint['optimizer']
    
    load_scheduler = checkpoint['scheduler']
    
    return load_model, load_optimizer, load_epoch, load_scheduler




def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    scaling = 256, 256
    image.thumbnail(scaling, Image.LANCZOS)
    
    crop = 224
    width, height = image.size
    
    left =(width - crop) / 2
    top = (height-crop) / 2
    right = left + crop
    bottom = top + crop
    
    image = image.crop((left, top, right, bottom))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = np.array(image) / 255.0
    
    image = (img_array - mean) / std
    
    image = image.transpose((2,0,1))
    
    return torch.from_numpy(image)




def predict(image_path, model, device, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    image = image.unsqueeze_(0)
    image = image.cuda().float()
      
    model.to(device)
    
    model.eval()
    
    with torch.no_grad():
        logps = model(image)
        top_p, idxs = torch.topk(logps, topk)
        
        idxs = np.array(idxs)
        idx_to_classes = {val:key for key, val in model.class_to_idx.items()}
        classes = [idx_to_classes[idx] for idx in idxs[0]]
        
        title = []
        for cls in classes:
            title.append(cat_to_name[str(cls)])
            
        return top_p, title
    
    
 

    
# load data from command line parser
cat_to_name_path = args.categories
device = args.gpu

if device == 'gpu':
    core = 'cuda'
else:
    core = 'cpu'

with open(cat_to_name_path, 'r') as f:
    cat_to_name = json.load(f)


load_model, load_optimizer, load_epoch, load_scheduler = load_checkpoint()

input_pic = args.image
topk = args.top_k

image_pil = Image.open(input_pic)
x, y = predict(image_pil, load_model, core, topk=topk)


for l in range(topk):
    print('Number: {}/{}..'.format(l+1, topk),\
          'Name: {}..'.format(y[l]),\
          'Probability: {:.3f}..%'.format(np.exp(x[0][l])*100))