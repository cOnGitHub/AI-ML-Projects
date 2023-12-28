# Imports
import argparse
import os
import json
import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image

def define_data_dirs(data_dir):
    '''
    Joins '\train', '\valid', and '\test' to the data dir.
    Returns dict of train, valid and test data dirs.
    '''
    data_dirs = {
        'train' : data_dir + '/train',
        'valid' : data_dir + '/valid',
        'test' : data_dir + '/test'
    }
    return data_dirs


def define_data_transforms():
    '''
    Composes transforms for train, valid, and test purposes.
    Crop size is 224, resize size is 256, normalization uses
    [0.485, 0.456, 0.406] and [0.229, 0.224, 0.225].
    Returns dict of transforms.
    '''
    data_transforms = {
        'train' : transforms.Compose([transforms.RandomRotation(330),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomVerticalFlip(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]),
        'valid_test' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    }
    return data_transforms

def load_image_datasets(data_dirs, data_transforms):
    '''
    Load the train, valid and test data sets from the data dirs
    using the data transforms.
    Returns dict of image datasets. 
    '''
    image_datasets = {
        'train' : datasets.ImageFolder(data_dirs['train'], transform=data_transforms['train']),
        'valid' : datasets.ImageFolder(data_dirs['valid'], transform=data_transforms['valid_test']),
        'test' : datasets.ImageFolder(data_dirs['test'], transform=data_transforms['valid_test'])
    }
    return image_datasets

def define_data_loaders(image_datasets):
    '''
    Defines train, valid and test data loaders from image datasets.
    Batch size i s64, shuffle is True.
    Returns dict of data loaders.
    '''
    data_loaders = {
        'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid' : torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True)
    }
    return data_loaders

def create_train_parser():
    '''
    Create parser for train.py.
    Required args: data_dir, arch, input_units, hidden_units, output_units
    Optional args with default: save_dir, learning_rate, drop_p, epochs
    Optional args: gpu
    '''
    parser = argparse.ArgumentParser('Parser for train.py')
    parser.add_argument('data_dir', action='store',
                        nargs=1,
                        help='Directory of the image data')
    parser.add_argument('--save_dir', action='store',
                        dest='save_dir',
                        default='/home/workspace/aipnd-project',
                        help='Directory for saving checkpoints')
    parser.add_argument('--arch', action='store',
                        dest='arch',
                        nargs=1,
                        help='Architecture of the model')
    parser.add_argument('--learning_rate', action='store', 
                        type=float,
                        default=0.001,
                        dest='learning_rate',
                        help='Learning rate')
    parser.add_argument('--input_units', action='store',
                        type=int,
                        nargs=1,
                        dest='input_size',
                        help='Size of input layer')
    parser.add_argument('--hidden_units', action='append', 
                        type=int,
                        nargs='+',
                        dest='hidden_sizes',
                        help='Multiple sizes of hidden layers')
    parser.add_argument('--output_units', action='store',
                        type=int,
                        nargs=1,
                        dest='output_size',
                        help='Size of output layer')
    parser.add_argument('--drop_p', action='store',
                        type=float,
                        default=0.5,
                        dest='drop_p',
                        help='Drop probability')
    parser.add_argument('--epochs', action='store', 
                        type=int,
                        default=3,
                        dest='epochs',
                        help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', 
                        default=False,
                        dest='use_gpu_mode',
                        help='Use gpu mode')
    return parser

def create_predict_parser():
    '''
    Create parser for predict.py.
    Required args: data_dir, arch, input_units, hidden_units, output_units
    Optional args with default: save_dir, learning_rate, drop_p, epochs
    Optional args: gpu
    '''
    parser = argparse.ArgumentParser('Parser for predict.py')
    # Use path to file as argument and not the file itself, for more information see here:
    # https://stackoverflow.com/questions/18862836/how-to-open-file-using-argparse
    parser.add_argument('path_to_image', action='store',
                        nargs=1,
                        help='Path to image')
    parser.add_argument('path_to_checkpoint', action='store',
                        nargs=1,
                        help='Path to checkpoint')
    parser.add_argument('--top_k', action='store',
                        type=int,
                        default=3,
                        help='Number of top probabilities')
    # Use a path to the file instead of the file itself, see comment above
    parser.add_argument('--category_names', action='store',
                        help='Path to category names file')
    parser.add_argument('--arch', action='store',
                        dest='arch',
                        nargs=1,
                        help='Architecture of the model')
    parser.add_argument('--gpu', action='store_true', 
                        default=False,
                        dest='use_gpu_mode',
                        help='Use gpu mode')
    return parser
    
    
def set_device(use_gpu_mode):
    if use_gpu_mode and torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        if use_gpu_mode:
            print('GPU mode not available, device is set to CPU')
        return torch.device('cpu')
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Image max size is required to be 256 x 256
    # Note that thumbnail is an inplace function
    image.thumbnail((256, 256))
    
    # Top left corner (16, 16) and bottom right corner (256-16=240, 256-16=240) depend on required size of (256, 256)
    image = image.crop((16,16,240,240))
    
    # Get ndarray from PIL
    np_image = np.array(image)
    
    # Move image from [0, 255] int to [0, 1] float as described here:
    # https://stackoverflow.com/questions/9974863/converting-a-0-255-integer-range-to-a-0-0-1-0-float-range
    np_image = [x / 255.0 for x in np_image]
        
    # Normalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Transpose image to be usable for the model
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image

def predict(image_path, model, device, topk=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implement the code to predict the class from an image file

    # Switch off gradients
    with torch.no_grad():
    
        # Get PIL image and the processed version of it
        image = Image.open(image_path)
        processed_image = process_image(image)

        # Convert ndarray to tensor
        # This idea and part of the code is from a solution by Kartik P. as stated on the Udacity forum
        torch_image = torch.from_numpy(processed_image)

        # Convert to float to avoid error message:
        # 'Expected object of type torch.DoubleTensor but found type torch.FloatTensor'
        torch_image = torch_image.float()

        # We need a forth dimension (though I have no idea why this is so), and so we use unsqueeze
        torch_image = torch_image.unsqueeze(0)

        # Solution from the Udacity forums to avoid following error message:
        # 'Expected object of type torch.cuda.FloatTensor but found type torch.FloatTensor'
        if (torch.cuda.is_available() and device=='cuda:0'):
            torch_image = torch_image.type(torch.cuda.FloatTensor)
            
        # Get model prediction
        outputs = model(torch_image)
        
        # As our model has log-softmax as output, we'll apply torch.exp(outputs) as described here:
        # https://discuss.pytorch.org/t/cnn-results-negative-when-using-log-softmax-and-nll-loss/16839
        outputs = torch.exp(outputs)

        # Get the top most k instances of the probabilities and the indices
        probs, im_indices = outputs.topk(topk)
        
        # Get the list of values out of the tensors
        # Convert Tensor to cpu as indicated here:
        # https://discuss.pytorch.org/t/convert-to-numpy-cuda-variable/499
        probs, im_indices = probs[0].cpu().numpy(), im_indices[0].cpu().numpy()

        # Extract the class_to_idx from model
        class_to_idx = model.class_to_idx

        # Invert the class_to_idx dictionary
        # Solution from https://stackoverflow.com/questions/483666/python-reverse-invert-a-mapping
        inv_class_to_idx = {idx: cls for cls, idx in class_to_idx.items()}

        # Get the classes from the indices, retain the same order as in probs
        im_classes = []
        for i in range(len(probs)):
            im_classes.append(inv_class_to_idx[im_indices[i]])
    
    return probs, im_classes

