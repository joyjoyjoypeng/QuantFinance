'''
Given a feature variable, x, and a corresponding response, y, write a python program called
local_linear.py to estimate the regression function f in the model y = f(x) + e using a local
linear kernel estimator and k-fold cross-validation to select the bandwidth. For simplicity,
you may assume that a Gaussian kernel should be used.

You should also provide functions for predicting new values and plotting the estimated regression
function. The required inputs to your program would be 2 files with n rows and 1 column of
numeric data - one containing the x variable and the other containing the y variable - an output
file path, and the number of folds. So at it's very basic level, the program could be invoked
as follows:

python local_linear.py --x xin.dms --y yin.dms --output output.dms --num_folds 10 

Optional arguments:
--plot True --xout xout.dms

And in output, you would write a n x 1 column of numeric data containing the values of the fitted
function evaluated at the points found in xin.

Your program should also support the optional arguments --plot, which would display a scatter plot
of y against x with the fitted function drawn over it, as well as --xout, to which the path to a
further n x 1 file of numeric data would be provided so that output would contains the values of
the fitted function evaluated at these points rather than the training points found in xin.
'''

import argparse
import math
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def file_import(file):
    '''
    This function imports a file and stores its contents in an array

    Inputs: 
        file (path): The file to be imported

    Returns:
        inputs (array): Array containing the values in the file
    '''

    with open(file,'r') as vals:
        inputs= np.array(list(map(lambda f: float(f.strip()), vals.readlines())))
    return inputs

def get_weight(x_training, x_actual, bandwidth_value):
    '''
    This function returns the weight of the x_actual in estimating the y value of the x_training,
    given a specific bandwitdth_value

    Inputs: 
        x_training (float): The x value that is being trained
        x_actual (float): The x value whose weight is being computed 
        bandwidth_value (float): The bandwidth being tested in a run

    Returns:
        weight_k (float): The k weight of the x_actual
    '''

    weight_k = math.exp(-((x_training - x_actual) ** 2) / bandwidth_value)
    return weight_k

bandwidth_values = np.linspace(0.1, 0.2, 2)

def get_lowest_h(x_train, y_train):
    '''
    This function tests all the possible bandwidth values and returns the one that produces the lowest
    MSE for a particular training set in a fold

    Inputs: 
        x_train (array): The x values being trained in a run
        y_train (array): The actual y values for the x values being trained

    Returns:
        lowest_h (float): The h value that caused the lowest MSE to be obtained
    '''

    mse_container = {}

    for h_val in bandwidth_values:
        errors = []
        for (z_index, z_val) in enumerate(x_train):
            k_vals = []
            y_vals = []
            for (x_index, x_val) in enumerate(x_train):
                if x_index != z_index:
                    k_vals.append(get_weight(z_val, x_val, h_val))
                    y_vals.append(y_train[x_index])

            y_hat = 0
            for (x_index, x_val) in enumerate(k_vals):
                weight = x_val / sum(k_vals)
                y_hat += weight * y_vals[x_index]

            error = y_hat - y_train[z_index]
            errors.append(error ** 2)

        mse = sum(errors)/len(errors)
        mse_container[h_val] = mse
        lowest_h = min(mse_container, key=mse_container.get)

    return lowest_h

def for_testing(x_test, y_test, h_val):
    '''
    This function is used to test the h value that had returned the lowest MSE during 
    the training phase against the x and y values in the testing set
    
    Inputs: 
        x_test (array): The x values being tested in a run
        y_test (array): The actual y values for the x values being tested
        h_val (float): The h value that had caused the lowest MSE to be obtained

    Returns:
        test_output (list): The h value that caused the lowest MSE to be obtained,
        as well as the MSE obtained by using that h value
    '''

    errors = []
    for (z_index, z_val) in enumerate(x_test):
        k_vals = []
        y_vals = []
        for (x_index, x_val) in enumerate(x_test):
            if x_index != z_index:
                k_vals.append(get_weight(z_val, x_val, h_val))
                y_vals.append(y_test[x_index])

        y_hat = 0
        for (x_index, x_val) in enumerate(k_vals):
            weight = x_val / sum(k_vals)
            y_hat += weight * y_vals[x_index]

        error = y_hat - y_test[z_index]
        errors.append(error ** 2)

    mse = sum(errors)/len(errors)
    test_output = [h_val, mse]
    
    return test_output

def get_optimal_h(the_x, the_y, k_folds):
    '''
    This function combines the previous two functions to obtain an optimal h
    across all the training and testing phases of the folds
    
    Inputs: 
        the_x (array): The entire set of x values
        the_y (array): The entire set of y values
        k_folds (int): The number of folds to be used for cross validation

    Returns:
        test_output (list): The h value that caused the lowest MSE to be obtained
    '''

    kf_creator = KFold(n_splits=k_folds, shuffle=True, random_state=np.random)
    tests = []
    counter = 0
    for train_index, test_index in kf_creator.split(the_x):
        counter += 1
        print("Training Fold", counter)
        x_train, x_test = the_x[train_index], the_x[test_index]
        y_train, y_test = the_y[train_index], the_y[test_index]

        h_val = get_lowest_h(x_train, y_train)
        result = for_testing(x_test, y_test, h_val)
        tests.append(result)

    print('Training Completed! :)')
    best_h = min(tests, key = lambda x: x[1])[0]
    #print(tests)
    return best_h

def actual_prediction(x_set, y_set, h_val):
    '''
    This function conducts the actual prediction on the x_set using the other
    y_set values and returns the predictions 
    
    Inputs: 
        x_set (array): The set of x values for which predictions need to be conducted
        y_set (array): The set of y values that will be used for predictions
        h_val (float): The optimal h that was found during the cross validation phase

    Returns:
        y_hats (list): The predictions of the y values in a list
    '''
    y_hats = []
    for (z_index, z_val) in enumerate(x_set):
        k_vals = []
        y_vals = []
        for (x_index, x_val) in enumerate(x_set):
            if x_index != z_index:
                k_vals.append(get_weight(z_val, x_val, h_val))
                y_vals.append(y_set[x_index])

        y_hat = 0
        for (x_index, x_val) in enumerate(k_vals):
            weight = x_val / sum(k_vals)
            y_hat += weight * y_vals[x_index]

        y_hats.append(y_hat)

    return y_hats

def create_output(pred, output):
    '''
    This function reates a new file given the predicted y values
    
    Inputs: 
        pred (array): The list of predicted y values
        output (path): The path for the output file to be created

    Returns:
        output (path): The path to the output file (For function testing purposes)
    '''

    with open(output, 'w') as new_file:
        for i in range(len(pred)-1):
            new_file.write(str(pred[i]) + "\n")
        new_file.write(str(pred[-1]))

    return output

def create_plot(x_vals, y_vals, pred):
    '''
    This function creates a plot showcasing the y_vals and predicted
    y values against the x_vals
    
    Inputs: 
        x_vals (array): The array of x values
        y_vals (array): The array of y values
        pred (list): The list of predicted y values

    Returns:
        --Nothing is returned as the plot will directly be created and saved in the folder--
    '''

    plt.scatter(x_vals, y_vals, color = 'lightgreen', s=30, alpha = 0.8,
                marker = 'o', label = "Actual Y")
    plt.scatter(x_vals, pred, color = 'red', s=30, alpha = 0.3,
                marker = '.',label= "Predicted Y")
    plt.title("Scatterplot of Predicted Y and Actual Y against X", loc='center')
    plt.legend()
    plt.savefig('Plot of Predictions.png')


def main (xin, yin, output, k_folds, plot = None, xout = None):
    '''
    This is the main function containing all the others
    
    Inputs: 
        xin (path): The path to the original x values
        yin (path): The path to the original y values
        output (path): The path foe the output file to be created
        k_folds (int): The number of folds to be used in the cross validation
        plot (bool): The optional command for whether a plot is to be generated
        xout (path): The optional path to a new set of x_values for prediction

    Returns:
        output (path): The path to the output file (For function testing purposes)
    '''
    x_inputs = file_import(xin)
    y_inputs = file_import(yin)
    bandwidth = get_optimal_h(x_inputs, y_inputs, k_folds)

    if xout:
        new_x_inputs = file_import(xout)
        y_prediction = actual_prediction(new_x_inputs, y_inputs, bandwidth)
    else:
        y_prediction = actual_prediction(x_inputs, y_inputs, bandwidth)

    output_file = create_output(y_prediction, output)

    if plot:
        if xout:
            create_plot(new_x_inputs, y_inputs, y_prediction)
        else:
            create_plot(x_inputs, y_inputs, y_prediction)
    return output_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x', required = True, help = "input file of x data")
    parser.add_argument('--y', required = True, help = "input file of y data")
    parser.add_argument('--output', required = True, help = "file path of the output")
    parser.add_argument('--num_folds', type = int, required=True, help = "number of k-folds")
    parser.add_argument('--plot', type = bool, help = "a plot of the input and prediction data")
    parser.add_argument('--xout', help = "file path to a set of different x for prediction")
    args = parser.parse_args()
    main(args.x,args.y,args.output,args.num_folds,args.plot,args.xout)
