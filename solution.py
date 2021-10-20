import matplotlib as plt
import os

import typing
from matplotlib import cm

import numpy as np
import torch
import gpytorch

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation

## Constant for Cost function
THRESHOLD = 35.5
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 5.0
COST_W_THRESHOLD = 20.0

"""
Fill in the methods of the Model. Please do not change the given methods for the checker script to work.
You can add new methods, and make changes. The checker script performs:
    M = Model()
    M.fit_model(train_x,train_y)
    prediction = M.predict(test_x)
It uses predictions to compare to the ground truth using the cost_function above.
"""

# In order to complete this task we have followed the GPyTorch Regression tutorial that can
# be found here : https://docs.gpytorch.ai/en/v1.1.1/examples/01_Exact_GPs/Simple_GP_Regression.html
# We have copied partially or in integrality some part of these tutorial/


class Model():
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
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = None

    def predict(self, x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param x: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """
        # Create a tensor of the test set
        test_x = torch.Tensor(x)

        # Turn the model into evaluation mode
        self.model.eval()
        self.likelihood.eval()

        # Get the prediction (in form of MutlvariateNormal)
        y_obs = self.likelihood(self.model(test_x))
        # For each observation we take the mean of this one as the prediction
        y_means = y_obs.mean.detach().numpy()
        y_preds = y_means

        # For each observation we also get the variance of the distribution
        variance_obs = y_obs.variance.detach().numpy()
        # Update prediction in order to be less penalized by the loss function
        #decision = np.zeros_like(y_preds)
        #for pollution under threshold with low confidence : don't underpredict
        #mask_1 = 
        #y_preds[(y_preds > THRESHOLD) & (cdf_half < confidence)] = THRESHOLD - epsilon

        return y_preds, y_means, variance_obs


    def fit_model(self, train_x: np.ndarray, train_y: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_x: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """

        # Construct tensors with the positive training data (and keep a copy of y_train for the loss function test)
        # copy_train_y = train_y
        data = np.column_stack((train_x,train_y))
        new_data = data[(data[:,0]>0) & (data[:,2]>0) & (data[:,2]>0)]
        train_x = new_data[:,:2]
        train_y = new_data[:,2]
        train_x = torch.Tensor(train_x[:12000])
        train_y = torch.Tensor(train_y[:12000])

        # Initialize the model with our training set and likelihood
        self.model = ExactGPModel(train_x, train_y, self.likelihood)

        # Train both the likelihood and the model in order to find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Use the Adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(100):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(train_x)
            # Compute loss and backpropagation gradients
            loss = -mll(output, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f  ' % (
                i + 1, 100, loss.item(),
            ))
            optimizer.step()

# Simplest form of Gaussian Model, Inference (cf. Tutorial)
class ExactGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = get_kernel("Matern-1/2")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def cost_function(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param y_true: Ground truth pollution levels as a 1d NumPy float array
    :param y_predicted: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
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



def get_kernel(kernel, composition="addition"):
    base_kernel = []
    if "RBF" in kernel:
        base_kernel.append(gpytorch.kernels.RBFKernel())
    if "linear" in kernel:
        base_kernel.append(gpytorch.kernels.LinearKernel())
    if "quadratic" in kernel:
        base_kernel.append(gpytorch.kernels.PolynomialKernel(power=2))
    if "Matern-1/2" in kernel:
        base_kernel.append(gpytorch.kernels.MaternKernel(nu=1/2))
    if "Matern-3/2" in kernel:
        base_kernel.append(gpytorch.kernels.MaternKernel(nu=3/2))
    if "Matern-5/2" in kernel:
        base_kernel.append(gpytorch.kernels.MaternKernel(nu=5/2))
    if "Cosine" in kernel:
        base_kernel.append(gpytorch.kernels.CosineKernel())

    if composition == "addition":
        base_kernel = gpytorch.kernels.AdditiveKernel(*base_kernel)
    elif composition == "product":
        base_kernel = gpytorch.kernels.ProductKernel(*base_kernel)
    else:
        raise NotImplementedError
    kernel = gpytorch.kernels.ScaleKernel(base_kernel)
    return kernel

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
    predicted_y, mean, var = model.predict(test_x)
    print('prediction ', predicted_y)
    print('mean ', mean)
    print('variance ', var)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
