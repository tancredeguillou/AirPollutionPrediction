import sys
import time

import numpy as np

import gpflow

import pandas as pd
import numpy as np
import tensorflow as tf
import math

from gpflow.utilities import print_summary
from gpflow.ci_utils import ci_niter

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def timeit(function):
    def wrapper(*args, **kwargs):
        start = time.time()
        retval = function(*args, **kwargs)
        stop = time.time()
        
        eprint(f"{function.__name__}(...) took {stop - start:.1f} seconds.")
        return retval

    return wrapper

## Constant for Cost function
THRESHOLD = 0.5
W1 = 1
W2 = 20
W3 = 100
W4 = 0.04


def cost_function(true, predicted):
    """
        true: true values in 1D numpy array
        predicted: predicted values in 1D numpy array

        return: float
    """
    cost = (true - predicted)**2

    # true above threshold (case 1)
    mask = true > THRESHOLD
    mask_w1 = np.logical_and(predicted>=true,mask)
    mask_w2 = np.logical_and(np.logical_and(predicted<true,predicted >=THRESHOLD),mask)
    mask_w3 = np.logical_and(predicted<THRESHOLD,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2
    cost[mask_w3] = cost[mask_w3]*W3

    # true value below threshold (case 2)
    mask = true <= THRESHOLD
    mask_w1 = np.logical_and(predicted>true,mask)
    mask_w2 = np.logical_and(predicted<=true,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2

    reward = W4*np.logical_and(predicted < THRESHOLD,true<THRESHOLD)
    if reward is None:
        reward = 0
    return np.mean(cost) - np.mean(reward)

"""
Fill in the methods of the Model. Please do not change the given methods for the checker script to work.
You can add new methods, and make changes. The checker script performs:


    M = Model()
    M.fit_model(train_x,train_y)
    prediction = M.predict(test_x)

It uses predictions to compare to the ground truth using the cost_function above.
"""

def run_adam(model, iterations, train_dataset, minibatch_size = 100):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    logf = []
    train_iter = iter(train_dataset.batch(minibatch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam()

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    for step in range(iterations):
        optimization_step()
        if step % 100 == 0:
            elbo = -training_loss().numpy()
            pad = len(str(iterations)) - 1
            eprint(f"Adam : step {step:>{pad}} of {iterations}, Elbo: {elbo}")
            logf.append(elbo)
    return logf

def select_inducing_variables(n, X):
    mask = X[:,0] >= 5
    X_left = X[~mask]
    random_indices = np.random.choice(X_left.shape[0], size=n)
    return np.concatenate([X[mask], X_left[random_indices]], axis=0)


kernels = { 
    "exp"       : gpflow.kernels.Exponential,
    "matern12"  : gpflow.kernels.Matern12, 
    "matern32"  : gpflow.kernels.Matern32,
    "matern52"  : gpflow.kernels.Matern52,
    "rbf"       : gpflow.kernels.SquaredExponential,
    "quadratic" : gpflow.kernels.RationalQuadratic,
    "cosine"    : gpflow.kernels.Cosine,
    "periodic"  : gpflow.kernels.Periodic,
    "poly"      : gpflow.kernels.Polynomial,
    "arccosine" : gpflow.kernels.ArcCosine,
    "coregion"  : gpflow.kernels.Coregion
} 

class Model():
    def __init__(self):
        params = Parameters()

        self.kernel = kernels[params.kernel]
        self.inducing_variables = params.inducing_variables
        self.adam_iterations = params.adam_iterations
        self.minibatch_size = params.minibatch_size
        self.predict_func = params.predict_func
        self.model = None

    @timeit
    def fit_model(self, train_x, train_y):
        train_y = np.expand_dims(train_y, axis=1) # check si necessaire
        train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).repeat().shuffle(train_x.shape[0])

        self.model = gpflow.models.SVGP(
            self.kernel(),
            gpflow.likelihoods.Gaussian(),
            inducing_variable = select_inducing_variables(n=self.inducing_variables, X=train_x),
            num_data = train_x.shape[0]
        )

        gpflow.set_trainable(self.model.inducing_variable, False)
        _ = run_adam(self.model, ci_niter(self.adam_iterations), train_dataset, self.minibatch_size)


    def predict(self, test_x):
        mean, var = self.model.predict_f(test_x)
        return mean.numpy()

class Parameters():
    kernel = "matern52"
    inducing_variables = 500
    adam_iterations = 3000
    minibatch_size = 50
    predict_func = lambda _, mean, var: mean

def main():
    print('Running solution.py')

    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Fit the model
    print('Fitting model')
    model = Model()
    model.fit_model(train_x, train_y)

    # Predict on the test features
    print('Predicting on test features')
    predicted_y = model.predict(test_x)
    print(predicted_y)


if __name__ == "__main__":
    main()
