import numpy as np
import os
from collections import OrderedDict
from torchvision import models
import torch
from torch import nn

def import_model(model_name, is_pretrained, device):
    '''
    Import the model as specified by model_name and by is_pretrained.
    List of model names: 'alexnet','vgg11','vgg13','vgg16',
    'vgg19','vgg11_bn','vgg13_bn','vgg16_bn','vgg19_bn','
    resnet18','resnet34','resnet50','resnet101','resnet152',
    'squeezenet1_0','squeezenet1_1','densenet121','densenet169',
    'densenet201','densenet161','inception_v3'
    '''
    # Import model
    model = getattr(models, model_name)(pretrained=is_pretrained)
    
    # Convert to device
    model.to(device)
    
    return model
        
def create_classifier(input_size, hidden_sizes, output_size, drop_p=0.5):
    '''
    Create classifier using input_size, hidden_sizes, output_size and drop_p.
    All layers except the output layer have ReLU, the output layer has LogSoftmax.
    All layers have dropout defined.
    '''
    # Create OrderedDict
    dict_of_layers = OrderedDict([])
    
    # Labels used
    fc_label = 'fc'
    relu_label = 'relu'
    dropout_label = 'dropout'
    
    # Fill OrderedDict with layers
    for i in np.arange(len(hidden_sizes) + 1):
        if i == 0:
            # Add input layer
            dict_of_layers[fc_label + str(i+1)] = nn.Linear(input_size, hidden_sizes[i])
            dict_of_layers[relu_label + str(i+1)] = nn.ReLU()
            dict_of_layers[dropout_label + str(i+1)] = nn.Dropout(p=drop_p)
        elif i == len(hidden_sizes):
            # Add output layer
            dict_of_layers[fc_label + str(i+1)] = nn.Linear(hidden_sizes[-1], output_size)
            dict_of_layers['output'] = nn.LogSoftmax(dim=1)
            dict_of_layers[dropout_label + str(i+1)] = nn.Dropout(p=drop_p)
        else:
            # Add hidden layers
            dict_of_layers[fc_label + str(i+1)] = nn.Linear(hidden_sizes[i-1], hidden_sizes[i])
            dict_of_layers[relu_label + str(i+1)] = nn.ReLU()
            dict_of_layers[dropout_label + str(i+1)] = nn.Dropout(p=drop_p)
        
    # Return classifier
    return nn.Sequential(dict_of_layers)

def validate_the_model(model, valid_loader, criterion, device):
    '''
    Validate the model on validation data with criterion and device.
    '''
    test_loss = 0
    accuracy = 0
    
    model.to(device)
    # Turn on evaluation mode
    model.eval()
    
    # Switch off usage of gradients to get better performance during validation
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            
            # Move images and labels to device
            images = images.to(device) 
            labels = labels.to(device)
            
            outputs = model.forward(images)
            
            # Calculate test loss and accuracy
            test_loss += criterion(outputs, labels).item()
            ps = torch.exp(outputs)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
            
    return test_loss, accuracy
    
def train_the_model(model, train_loader, valid_loader, epochs, print_every, criterion, optimizer, device):
    '''
    Train the model on training data using criterion, optimizer and device. 
    Train number of epochs and print every number of steps according to parameter input. 
    '''
    steps = 0
    
    # change to device
    model.to(device)
    
    for e in range(epochs):
        model.train()
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                # Put model into eval mode
                model.eval()
                
                valid_loss, accuracy = validate_the_model(model, valid_loader, criterion, device)
                
                print("Epoch: {}/{}: ".format(e+1, epochs),
                      "Train Loss: {:.4f}".format(running_loss/print_every),
                      "Valid Loss: {:.4f}".format(valid_loss/len(valid_loader)),
                      "Validation accuracy: {:.4f}%".format(100*accuracy/len(valid_loader)))
                
                running_loss = 0
                
                # Put model back into train mode
                model.train()

def save_checkpoint(input_size, output_size, hidden_sizes, state_dict, optim_state_dict, epochs, class_to_idx, classifier, save_dir, arch):
    '''
    Create checkpoint including input_size, output_size, hidden_sizes, state_dict, optim_state_dict, epochs, class_to_idx, classifier, and arch
    Note that the architecture will be appended to the filename, for example, checkpoint_vgg16.pth with path equal to 'vgg16'
    '''
    # Create the checkpoint
    checkpoint = {
        'input_size' : input_size,
        'output_size' : output_size,
        'hidden_sizes' : hidden_sizes, 
        'state_dict' : state_dict,
        'optimizer_state_dict' : optim_state_dict, 
        'epochs' : epochs,
        'class_to_idx' : class_to_idx,
        'classifier' : classifier,
        'arch' : arch}

    # Create filepath for checkpoint
    file_name = 'checkpoint_' + arch + '.pth'
    filepath = os.path.join(save_dir, file_name)
    
    # Save the checkpoint
    torch.save(checkpoint, filepath)
    print('\nSave checkpoint to {}'.format(filepath))

def rebuild_model(path_to_checkpoint, arch, device):
    '''
    Load the checkpoint and the model.
    '''
    # Load the checkpoint
    file_name = 'checkpoint_' + arch + '.pth'
    checkpoint_file = os.path.join(path_to_checkpoint, file_name)
    loaded_checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
    
    # Import the model
    is_model_pretrained = True
    loaded_model = import_model(arch, is_model_pretrained, device)
    
    loaded_model.classifier = loaded_checkpoint['classifier']
    loaded_model.load_state_dict(loaded_checkpoint['state_dict'])
    loaded_model.class_to_idx = loaded_checkpoint['class_to_idx']

    return loaded_model