from __future__ import print_function

import argparse
import os
import pandas as pd
import numpy as np

# sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23. 
# from sklearn.externals import joblib
# Import joblib package directly
import joblib

## TODO: Import any additional libraries you need to define a model
### Solution 2: Adaboost
#from sklearn.ensemble import AdaBoostClassifier
### Solution 3: Neural Net MLPClassifier
#from sklearn.neural_network import MLPClassifier
### Solution 4: Nearest Neighbors
#from sklearn.neighbors import KNeighborsClassifier
### Solution 5: Decision Tree
#from sklearn.tree import DecisionTreeClassifier

# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    # joblib replaced by numpy.load due to error on executing train.py, see https://knowledge.udacity.com/questions/342430
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    #model = np.load(os.path.join(model_dir, "model.npy"))
    print("Done loading model.")
    
    return model


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    ## TODO: Add any additional arguments that you will need to pass into your model
    ### Solution 2: Adaboost with default parameters n_estimators=50, learning_rate=1.0
    #parser.add_argument('--n_estimators', type=int, default=50)
    #parser.add_argument('--learning_rate', type=float, default=1.0)
    ### Solution 3: Neural Net MLPClassifier
    #parser.add_argument('--hidden_layer_sizes', type=str, default='100')
    #parser.add_argument('--activation', type=str, default='relu')
    #parser.add_argument('--solver', type=str, default='adam')
    #parser.add_argument('--learning_rate_init', type=float, default=0.001)    
    ### Solution 4: Nearest Neighbors
    #parser.add_argument('--n_neighbors', type=int, default=5)
    #parser.add_argument('--weights', type=str, default='uniform')
    #parser.add_argument('--algorithm', type=str, default='auto')
    #parser.add_argument('--leaf_size', type=int, default=30)    
    ### Solution 5: Decision Tree
    #parser.add_argument('--criterion', type=str, default='gini')
    #parser.add_argument('--splitter', type=str, default='best')
    #parser.add_argument('--max_depth', type=int, default=None)

    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    
    
    
    ## --- Your code here --- ##
    

    ## TODO: Define a model 
    ### Solution 2: Adaboost
    #model = AdaBoostClassifier(n_estimators=args.n_estimators, learning_rate=args.learning_rate)
    ### Solution 3: Neural Net MLPClassifier
    #model = MLPClassifier(hidden_layer_sizes=tuple(list(map(int, args.hidden_layer_sizes.split()))), 
    #                      activation=args.activation, 
    #                      solver=args.solver, 
    #                      learning_rate_init=args.learning_rate_init)
    ### Solution 4: Nearest Neighbors
    #model = KNeighborsClassifier(n_neighbors=args.n_neighbors,
    #                            weights=args.weights,
    #                            algorithm=args.algorithm, 
    #                            leaf_size=args.leaf_size)
    ### Solution 5: Decision Tree
    #model = DecisionTreeClassifier(criterion=args.criterion, splitter=args.splitter, max_depth=args.max_depth)
    
    ## TODO: Train the model
    ### Solution 2 Adaboos, 3 Neural Net MLPClassifier, 4 Nearest Neighbors, 5 Decision Tree
    #model.fit(train_x, train_y)
    
    
    ## --- End of your code  --- ##
    

    # Save the trained model
    # joblib replaced by numpy.load due to error on executing train.py, see https://knowledge.udacity.com/questions/342430
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    #np.save(model, os.path.join(args.model_dir, "model.npy"))
