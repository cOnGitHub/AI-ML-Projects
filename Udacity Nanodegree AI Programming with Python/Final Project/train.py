# Imports
import torch
from torch import nn
from torch import optim
import argparse
import os
import workspace_utils
from workspace_utils import active_session
import model_lib
import project_utils

# Global variables
print_every = 40

# Example commands
# vgg16 with gpu
#python train.py /home/workspace/aipnd-project/flowers --save_dir /home/workspace/aipnd-project/checkpoints --arch vgg16 --learning_rate 0.001 --input_units 25088 --hidden_units 4096 1024 --output_units 102 --drop_p 0.35 --epochs 3 --gpu

# vgg16 without gpu
#python train.py /home/workspace/aipnd-project/flowers --save_dir /home/workspace/aipnd-project/checkpoints --arch vgg16 --learning_rate 0.001 --input_units 25088 --hidden_units 4096 1024 --output_units 102 --drop_p 0.35 --epochs 3

# densenet121 with gpu
#python train.py /home/workspace/aipnd-project/flowers --save_dir /home/workspace/aipnd-project/checkpoints --arch densenet121 --learning_rate 0.001 --input_units 1024 --hidden_units 512 --output_units 102 --drop_p 0.35 --epochs 3 --gpu

# alexnet with gpu
#python train.py /home/workspace/aipnd-project/flowers --save_dir /home/workspace/aipnd-project/checkpoints --arch alexnet --learning_rate 0.001 --input_units 9216 --hidden_units 4096 1024 --output_units 102 --drop_p 0.4 --epochs 3 --gpu

# Create parser
parser = project_utils.create_train_parser()
print(parser.parse_args())
train_args = parser.parse_args()

# Assign args to variables
data_dir = train_args.data_dir[0]
save_dir = train_args.save_dir
arch = train_args.arch[0]
learning_rate = train_args.learning_rate
input_size = train_args.input_size[0]
hidden_sizes = train_args.hidden_sizes[0]
print(hidden_sizes)
output_size = train_args.output_size[0]
drop_p = train_args.drop_p
epochs = train_args.epochs
use_gpu_mode = train_args.use_gpu_mode

# Set device
device = project_utils.set_device(use_gpu_mode)
print("device is: {}".format(device))

# To avoid error 'Found no NVIDIA driver on your system. Please check that you
# have an NVIDIA GPU and installed a driver from' when running on CPU 
# I have added the next line of code similar as mentioned here:
# https://github.com/pytorch/text/issues/236
if not torch.cuda.is_available() and torch.device is None: 
    torch.device = -1

# Init data
data_dirs =  project_utils.define_data_dirs(data_dir)
data_transforms =  project_utils.define_data_transforms()
image_datasets = project_utils.load_image_datasets(data_dirs, data_transforms)
data_loaders = project_utils.define_data_loaders(image_datasets)

# Create model
is_model_pretrained = True
model = model_lib.import_model(arch, is_model_pretrained, device)

# freeze the parameters of the features in the model
for param in model.parameters():
    param.requires_grad = False

# Create classifier
classifier = model_lib.create_classifier(input_size, hidden_sizes, output_size, drop_p)
print('New classifier created:')
print(classifier)

# Replace classifier of model
print('Replace classifier of model')
model.classifier = classifier

# Convert to device
model.to(device)

# Set the criterion
criterion = nn.NLLLoss()
# Convert criterion to device
criterion.to(device)

# train only the classifier of the model
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# Train the model
# Keep session open during training
with active_session():
    model_lib.train_the_model(model, data_loaders['train'], data_loaders['valid'], epochs, print_every, criterion, optimizer, device)

# Validate the model on the test set
test_loss, test_accuracy = model_lib.validate_the_model(model, data_loaders['test'], criterion, device)
print('The testing accuracy of the network is: %d %%' % (100*test_accuracy/len(data_loaders['test'])))

# Save  checkpoint of model
model_lib.save_checkpoint(input_size, output_size, hidden_sizes, model.state_dict(), optimizer.state_dict, epochs, image_datasets['train'].class_to_idx, classifier, save_dir, arch)
    
