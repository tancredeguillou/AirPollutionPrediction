import time

import matplotlib as plt
import os
import typing

from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib import cm

import sys

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import gpflow
from gpflow.ci_utils import ci_niter

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
gpflow.config.set_default_summary_fmt("notebook")
# convert to float64 for tfp to play nicely with gpflow in 64
f64 = gpflow.utilities.to_default_float



# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation


# Cost function constants
THRESHOLD = 35.5
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 5.0
COST_W_THRESHOLD = 20.0

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
    mask = X[:,0] > 1
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

class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        params = Parameters()
        self.kernel = kernels[params.k]
        self.inducing_variables = params.inducing_variables
        self.adam_iterations = params.adam_iterations
        self.minibatch_size = params.minibatch_size
        self.predict_func = params.predict_func
        self.model = None


    def predict(self, x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param x: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        # TODO: Use your GP to estimate the posterior mean and stddev for each location here
        gp_mean = np.zeros(x.shape[0], dtype=float)
        gp_std = np.zeros(x.shape[0], dtype=float)

        # TODO: Use the GP posterior to form your predictions here
        predictions = self.model.predict_y(x)
        mean, var = self.model.predict_f(x)

        return tf.reshape(predictions[:][0], [-1]), tf.reshape(predictions[:][1], [-1]), tf.reshape(var, [-1])


    def fit_model(self, train_x: np.ndarray, train_y: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_x: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """

        train_y = np.expand_dims(train_y, axis=1) # check si necessaire
        train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).repeat().shuffle(train_x.shape[0])

        k1 = gpflow.kernels.RationalQuadratic()
        k2 = gpflow.kernels.RBF()
        k = k1 + k2

        self.model = gpflow.models.SVGP(
            k,
            gpflow.likelihoods.Gaussian(),
            inducing_variable = select_inducing_variables(n=self.inducing_variables, X=train_x),
            num_data = train_x.shape[0]
        )

        gpflow.set_trainable(self.model.inducing_variable, False)
        _ = run_adam(self.model, ci_niter(self.adam_iterations), train_dataset, self.minibatch_size)


def cost_function(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param y_true: Ground truth pollution levels as a 1d NumPy float array
    :param y_predicted: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    print(y_true.shape)
    print(y_predicted.shape)
    assert y_true.ndim == 1 and y_predicted.ndim == 1 and y_true.shape == y_predicted.shape

    # Unweighted cost
    cost = (y_true - y_predicted) ** 2
    weights = np.zeros_like(cost)

    # Case i): overprediction
    mask_1 = y_predicted > y_true
    weights[mask_1] = COST_W_OVERPREDICT

    # Case ii): true is above threshold, prediction below
    mask_2 = (y_true >= THRESHOLD) & (y_predicted < THRESHOLD)
    weights[mask_2] = COST_W_THRESHOLD

    # Case iii): everything else
    mask_3 = ~(mask_1 | mask_2)
    weights[mask_3] = COST_W_NORMAL

    # Weigh the cost and return the average
    return np.mean(cost * weights)

class Parameters():
    k = "poly"
    inducing_variables = 50
    adam_iterations = 5000
    minibatch_size = 100
    predict_func = lambda _, mean, var: mean

def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle('Extended visualization of task 1')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.predict(visualization_xs)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title('Predictions')
    fig.colorbar(predictions_plot)

    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection='3d')
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=cm.get_cmap()(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title('GP means, colors are GP stddev')

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title('GP estimated stddev')
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def main():
    # Load the training dateset and test features
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
    print('prediction ', predicted_y[:][0])
    print('mean ', predicted_y[:][1])
    print('variance ', predicted_y[1])


    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
