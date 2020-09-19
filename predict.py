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

parser = argparse.ArgumentParser("arguments for the predict.py file")


parser.add_argument("checkpoint_path", type=str, help="The checkpoint")
parser.add_argument("image_path", type=str, help="The image to predict")
parser.add_argument("--labels", type=str, default="cat_to_name.json", help="The labels that are associated with the images")
parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")

args = parser.parse_args()

with open(args.labels, 'r') as f:
    labels = json.load(f)


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    model = checkpoint["model"]

    optimizer = checkpoint["optimizer_state_dict"]

    model.class_to_idx = checkpoint["class_to_idx"]

    return model, checkpoint, optimizer, model.class_to_idx

# Image Processing

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    # original size
    width, height = im.size
    if width > height:
        height = 256
        im.thumbnail((50000, height), Image.ANTIALIAS)
    else:
        width = 256
        im.thumbnail((width, 50000), Image.ANTIALIAS)

    # set new size
    width, height = im.size
    reduce_size = 224
    left = (width - reduce_size) / 2
    top = (height - reduce_size) / 2
    right = left + reduce_size
    bottom = top + reduce_size

    im = im.crop((left, top, right, bottom))

    np_image = np.array(im)
    np_image = np_image / 255

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    np_image = (np_image - means) / stds
    py_np_image = np_image.transpose(2, 0, 1)
    return py_np_image

# show image
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    return ax

# Predict

def predict(image_path, model, device=args.device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    with torch.no_grad():
        image = process_image(image_path)
        image = torch.Tensor(image)
        image = image.to(device)
        image.unsqueeze_(0)
        image = image.float()
        model, checkpoint, optimizer, class_to_idx = load_checkpoint(model)
        model.to(device)
        outputs = model(image)

        probs, classes = torch.exp(outputs).topk(topk)

        return probs[0].tolist(), classes[0].add(1).tolist()





image_path = args.image_path
model = args.checkpoint_path
probs, indexes = predict(image_path, model)
model, checkpoint, optimizer, class_to_idx = load_checkpoint(model)
reversed_dictionnary = {v:k for k,v in class_to_idx.items()}
classes = [reversed_dictionnary[index] for index in indexes]
names = [labels[str(classic)] for classic in classes]




print(class_to_idx)


reversed_dictionnary = {v:k for k,v in class_to_idx.items()}
print(reversed_dictionnary)

classes = [reversed_dictionnary[index] for index in indexes]
print(classes)



names = [labels[str(classic)] for classic in classes]
print(names)
print(probs)

print("The most probable class is {} with a probability of {}".format(names[0], max(probs)))

# all_names = [labels[str(i)] for i in classes]

# print(all_names)
# print(probs)