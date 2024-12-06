"""Module holding utilities for study of bias and varianve and the comparison or 
regression models past `interpolation` point.
"""
from typing import Callable

from scipy.optimize import minimize
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from tqdm.notebook import tqdm

def sample_data(
    fn: Callable, 
    num_samples=10, 
    noise_std=1, 
    min_of_interval=-1,
    max_of_interval=1,
    prob_noise_exists=1.0,
    test_set=False,

):
    """_summary_

    Args:
        fn (Callable): Fn to generate outputs
        num_sample (int, optional): How many training samples to generate. 
            Defaults to 10.
        noise_std (int, optional): Variance of the normal distribution generating the 
            noise. Defaults to 1.
        min_of_interval (int, optional): Lower bound of the interval defining the 
            support. Defaults to -1.
        max_of_interval (int, optional): Upper bound of the interval defining the 
            support. Defaults to 1.
        prob_noise_exists (float, optional): Probability that an instance has associate 
            noise, i.e. is a noisy observation. Defaults to 1.0.
        test_set (bool, optional): Whether to generate data for a test set. The 
            difference in this case is that the generated samples won't be random 
            uniform samples but evenly spaced points across the specified interval 
            instead. Defaults to False.

    Returns:
        _type_: _description_
    """
    if test_set:
        x = np.linspace(min_of_interval, max_of_interval, num_samples)
    else:
        x = np.random.uniform(min_of_interval, max_of_interval, num_samples)
    y = (
        fn(x) + 
        noise_std * np.random.binomial(
            1, prob_noise_exists, num_samples
        ) * np.random.randn(num_samples)
    )
    return x, y


def compute_average_predictions(clf_predictions):
    """`clf_predictions` assumed to be `k x N` array for `k` 
    classifier predictions on `N` points

    Args:
        clf_predictions (_type_): _description_

    Returns:
        _type_: _description_
    """
    return clf_predictions.mean(axis=0)


def compute_bias(
    average_prediction, 
    true_labels, 
    loss="l2"
):
    if loss == "l2":
        return ((average_prediction - true_labels)**2).mean()
    else:
        raise ValueError(f"[{loss}] loss not supported.")

# TODO: Do we want to consider the case where we compute the variance per classifier 
# to show decomposed error per classifier?
def compute_variance(
    clf_predictions, 
    average_predictions,
    loss="l2",
):
    if loss == "l2":
        av_loss = (clf_predictions - average_predictions)**2
        return av_loss.mean()
    else: 
        raise ValueError(f"[{loss}] loss not supported.")


def fit_poly_model_closed_form(x_train, y_train, degree=0):
    model = Pipeline([
        ("poly_transformer", PolynomialFeatures(degree=degree, include_bias=True)),
        ("reg", linear_model.LinearRegression(fit_intercept=True))
        ])
    model.fit(x_train.reshape(-1, 1), y_train)
    
    return model


def regression_model(x, w):
        return np.dot(x, w)


def fit_lin_reg_scipy(
    X, 
    y_train, 
    w0=None,
    optimization_method="BFGS"
):

    def loss(w, x, ys):
        predictions = regression_model(x, w)
        return ((predictions - ys)**2).sum()
    
    if w0 is None:
        w0 = np.random.randn(X.shape[1])
    
    result = minimize(
        loss, w0, (X, y_train), method=optimization_method
    )
    return result


# TODO: Add ability to modify variance at initialisation.
def fit_mlp_model(
    x_train, 
    y_train, 
    hidden_layer_sizes=(100,),
    activation="tanh",
    solver="lbfgs",
    max_iter=5000,
):
    model = MLPRegressor(
        activation=activation, 
        solver=solver, 
        alpha=0, 
        hidden_layer_sizes=hidden_layer_sizes, 
        max_iter=max_iter
    )
    model.fit(x_train.reshape(-1, 1), y_train)
    return model


def tanh_features(x, min, max, n_features):
    return np.tanh(
        x.reshape(-1, 1) - np.linspace(min, max, n_features)
    )


def gauss_features(x, min, max, n_features, ls=1):
    return np.exp(
        -(x.reshape(-1, 1) - np.linspace(min, max, n_features))**2/ls
    )


def run_experiment(
    x_test,
    y_test,
    model_type="poly", 
    n_epochs=50, 
    degrees=99,
    sampling_fn_kwargs=None,
    feature_fn=None,
):
    print(f"Fitting [{model_type}] regression")
    sampling_kwargs = {
        "num_samples": 10, 
        "noise_std": 0.2, 
        "min_of_interval": -1,
        "max_of_interval": 1,
        "prob_noise_exists": 1.0,
        "test_set": False,
    }
    if sampling_fn_kwargs is not None:
        sampling_kwargs.update(**sampling_fn_kwargs)

    if isinstance(degrees, int):
        degrees = [*range(degrees)]
    increasing_complexity_data = []
    for i, d in tqdm(enumerate(degrees)): #, unit="degree", total=len(degrees)):
        deg_specific_data = {}
        predictions = []
        for e in tqdm(range(n_epochs)):#, unit="epoch", total=len(n_epochs)):
            x_train, y_train = sample_data(**sampling_kwargs)
            if model_type == "poly":
                model = fit_poly_model_closed_form(x_train, y_train, degree=d)
                predictions.append(
                    model.predict(x_test.reshape(-1, 1))
                )
            elif model_type == "mlp":
                model = fit_mlp_model(x_train, y_train, hidden_layer_sizes=(d + 1,))
                predictions.append(
                    model.predict(x_test.reshape(-1, 1))
                )
            elif model_type == "scipy_reg":
                if not callable(feature_fn):
                    raise ValueError("Feature function must be a callable.")
                X = feature_fn(x_train, -1, 1, d + 1)
                results = fit_lin_reg_scipy(X, y_train)
                predictions.append(regression_model(
                    feature_fn(x_test, -1, 1, d + 1), results.x
                ))
            else:
                raise ValueError(f"[{model_type}] not supported")
            
        predictions = np.asarray(predictions)
        deg_specific_data["predictions"] = predictions
        av_predictions = compute_average_predictions(predictions)
        deg_specific_data["av_predictions"] = av_predictions
        deg_specific_data["var"] = compute_variance(predictions, av_predictions)
        deg_specific_data["bias2"] = compute_bias(av_predictions, y_test)
        increasing_complexity_data.append(deg_specific_data)
    return increasing_complexity_data