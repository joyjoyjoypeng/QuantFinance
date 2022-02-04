'''
Given a feature variable, x, and a corresponding response, y, write a python program called
local_linear.py to estimate the regression function f in the model y = f(x) + e using a local
linear kernel estimator and k-fold cross-validation to select the bandwidth. For simplicity,
you may assume that a Gaussian kernel should be used.

You should also provide functions for predicting new values and plotting the estimated regression
function. The required inputs to your program would be 2 files with n rows and 1 column of
numeric data – one containing the x variable and the other containing the y variable – an output
file path, and the number of folds. So at it's very basic level, the program could be invoked
as follows:

python local_linear.py --x xin –-y yin –-output output –-num_folds 10

And in output, you would write a n x 1 column of numeric data containing the values of the fitted
function evaluated at the points found in xin.

Your program should also support the optional arguments –-plot, which would display a scatter plot
of y against x with the fitted function drawn over it, as well as –-xout, to which the path to a
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
    Imports a file and stores it as an array
    '''
    with open(file,'r') as vals:
        inputs= np.array(list(map(lambda f: float(f.strip()), vals.readlines())))
    return inputs

def get_weight(x_training, x_actual, bandwidth_value):
    '''
    Gets weight
    '''
    weight_k = math.exp(-((x_training - x_actual) ** 2) / bandwidth_value)
    return weight_k

bandwidth_values = np.linspace(0.1, 0.2, 2)

def get_lowest_h(x_train, y_train):
    '''
    This gives us the lowest h for a particular training set in a fold
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

    return min(mse_container, key=mse_container.get)

def for_testing(x_test, y_test, h_val):
    '''
    This function is used to test the h with the lowest MSE in the training phase
    against the testing set
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
    return [h_val, mse]

def get_optimal_h(the_x, the_y, k_folds):
    '''
    Combines the previous two functions to obtain an optimal h
    across all the folds
    '''
    kf_creator = KFold(n_splits=k_folds, shuffle=True, random_state=np.random) # Change to np.random later
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
    best_h = min(tests, key = lambda x: x[1])
    #print(tests)
    return best_h[0]

def actual_prediction(x_set, y_set, h_val):
    '''
    Conducts the actual prediction on the x_set using the other
    y_set values and returns the predictions as an array
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
    Creates a new file given the predicted y values
    '''
    with open(output, 'w') as new_file:
        for i in range(len(pred)-1):
            new_file.write(str(pred[i]) + "\n")
        new_file.write(str(pred[-1]))
    return output

def create_plot(x_vals, y_vals, pred):
    '''
    Creates a plot
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
    Main
    return something for easier testing
    '''
    x_inputs = file_import(xin)
    y_inputs = file_import(yin)
    bandwidth = get_optimal_h(x_inputs, y_inputs, k_folds)

    if xout:
        new_x_inputs = file_import(xout)
        y_prediction = actual_prediction(new_x_inputs, y_inputs, bandwidth)
    else:
        y_prediction = actual_prediction(x_inputs, y_inputs, bandwidth)

    result = create_output(y_prediction, output)

    if plot:
        if xout:
            create_plot(new_x_inputs, y_inputs, y_prediction)
        else:
            create_plot(x_inputs, y_inputs, y_prediction)
    return result


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
