import matplotlib.pyplot as plt
import pandas as pd
import torch
import json
import os
import project_utils
import model_lib

# Example commands

# vgg16 with gpu
#python predict.py flowers/test/12/image_04016.jpg checkpoints --top_k 5 --category_names cat_to_name.json --arch vgg16 --gpu

# vgg16 without gpu
#python predict.py flowers/test/12/image_04016.jpg checkpoints --top_k 5 --category_names cat_to_name.json --arch vgg16

# densenet121 without gpu
#python predict.py flowers/test/12/image_04016.jpg checkpoints --top_k 6 --category_names cat_to_name.json --arch densenet121

# alexnet without gpu
#python predict.py flowers/test/12/image_04016.jpg checkpoints --top_k 4 --category_names cat_to_name.json --arch alexnet

# Create parser
parser = project_utils.create_predict_parser()
args = parser.parse_args()
print('\nargs:')
print(args)

# Set device
device = project_utils.set_device(args.use_gpu_mode)
print("\ndevice is: {}".format(device))
if not torch.cuda.is_available() and torch.device is None: 
    torch.device = -1

# Load the model
loaded_model = model_lib.rebuild_model(args.path_to_checkpoint[0], args.arch[0], device)
loaded_model.to(device)

# Get prediction
probs, classes = project_utils.predict(args.path_to_image[0], loaded_model, device, args.top_k)

# Get categories using cat_to_name file
categories = []
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
for i in range(len(classes)):
    categories.append(cat_to_name[classes[i]])

# Display data frame of top k probabilities 
items = {'Top {} probabilities'.format(args.top_k) : pd.Series(probs),
         'Categories' : pd.Series(categories)}
df = pd.DataFrame(items)
print('\n' + str(df))

