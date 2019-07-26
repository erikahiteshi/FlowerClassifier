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
from collections import OrderedDict
from torch.autograd import Variable

def load_checkpoint(args):
    checkpoint = torch.load(args.saved_model)
    learning_rate = checkpoint['learning_rate']
    class_to_idx = checkpoint['class_to_idx']

#    model = models.densenet121(pretrained=True);
    model_name = checkpoint['arch']
   
    if model_name == 'densenet121':
       model = models.densenet121(pretrained=True)

    else:
        print("model is not available")
   
    for param in model.parameters(): 
        param.requires_grad = False
        
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    print(model.classifier)
    if args.use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(device)
    print("checkpoint loaded")
    return model

def process_image(args):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    expects_means = [0.485, 0.456, 0.406]
    expects_std = [0.229, 0.224, 0.225]
    print(type(args.image_path))
    pil_image = Image.open(args.image_path)
    
    np_image = np.array(pil_image)
   
    imagetransforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(expects_means, expects_std)])
    pil_image = imagetransforms(pil_image)
    return pil_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def read_category_names(args):
    with open(args.category_names, 'r') as f:
        loaded_json = json.load(f)
    return loaded_json

def predict(args, model, loaded_json):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implemented the code to predict the class from an image file
    # Evaluation mode - we don't want dropout included in this
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # load image as torch.Tensor using the process_image function
    image = process_image(args)
    pytorch_tensor = torch.tensor(image)
    pytorch_tensor = pytorch_tensor.float()
    pytorch_tensor_image = pytorch_tensor.unsqueeze(0)   
    inputimage = Variable(pytorch_tensor_image)
    inputimage = inputimage.to(device)
    # Disabling gradient calculation as this is not required in evaluation mode
    with torch.no_grad():
        
        #get the output of the model
        output = model.forward(inputimage)
        
        #get the probabilities
        ps = torch.exp(output)
        
        #returning the highest k probabilities
        top_probs, top_labels = ps.topk(args.topk)
        
        #separate our top predictions/labels into a list
       # top_predictions = torch.exp(output).data
       # top_predictions = top_probs.detach().numpy().tolist()
        
        top_labels = top_labels.tolist()
        top_predictions = top_probs.tolist()
        print(top_predictions)
        print(top_labels)
        
        #Put these lists into a dataframe and link with our json file which has the names of flowers
        #match up our class outputs to the names in the json
        classprediction = pd.DataFrame({'class':pd.Series(model.class_to_idx),'flower_name':pd.Series(loaded_json)})
        classprediction = classprediction.set_index('class')
        
        #Now link this df with our toplabels & top predictions
        classprediction = classprediction.iloc[top_labels[0]]
        classprediction['predictions'] = top_predictions[0]
        print(classprediction)
        return classprediction

def main():

    parser = argparse.ArgumentParser(description='Flower Classification Predictor')
    parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu or not')
    parser.add_argument('--image_path', type=str, default ="flowers/test/100/image_07896.jpg", help="path of image")
    parser.add_argument('--category_names', type=str, default="cat_to_name.json", help="PATH to file - converts category label to name")
    parser.add_argument('--hidden_units', type=int, default=512, help='hidden units for fc layer')
    parser.add_argument('--saved_model' , type=str, default='densenet121_checkpoint.pth', help='PATH of my model')
    parser.add_argument('--topk', type=int, default=5, help='Number of top probabilities to show')
    args = parser.parse_args()
    model = load_checkpoint(args)
    print(args.image_path)
    image = args.image_path
    
    loaded_json = read_category_names(args)
    predict(args, model, loaded_json)
    

if __name__ == '__main__':
    main()