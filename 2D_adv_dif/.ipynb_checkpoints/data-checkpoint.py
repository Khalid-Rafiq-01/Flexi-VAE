import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from dataclasses import dataclass, asdict
import json


# Rnum = 1000
num_time_steps = 500

def get_dt(num_time_steps):
    return 2.0/num_time_steps

dt = get_dt(num_time_steps)


def exact_solution(alpha, t, L=2.0, Nx=128, Ny=128, c=1.0):
    nu = 1.0 / alpha
    x_vals = np.linspace(-L, L, Nx)
    y_vals = np.linspace(-L, L, Ny)
    X, Y = np.meshgrid(x_vals, y_vals)
    if t <= 0:
        return np.zeros_like(X)
    rx = X - c * t
    ry = Y
    r2 = rx**2 + ry**2
    denominator = 4.0 * nu * t
    amplitude = 1.0 / (4.0 * np.pi * nu * t)
    U = amplitude * np.exp(-r2 / denominator)
    return U


class AdvectionDiffussionDataset:
    def __init__(self, 
                X: np.ndarray = None,
                X_tau: np.ndarray = None,
                t_values: np.ndarray = None,
                tau_values: np.ndarray = None,
                alpha_values: np.ndarray = None):
        self.X = X
        self.X_tau = X_tau
        self.t_values = t_values
        self.tau_values = tau_values
        self.alpha_values = alpha_values

    def append(self, other):
        self.X = np.concatenate([self.X, other.X]) if self.X is not None else other.X
        self.X_tau = np.concatenate([self.X_tau, other.X_tau]) if self.X_tau is not None else other.X_tau
        self.t_values = np.concatenate([self.t_values, other.t_values]) if self.t_values is not None else other.t_values
        self.tau_values = np.concatenate([self.tau_values, other.tau_values]) if self.tau_values is not None else other.tau_values
        self.alpha_values = np.concatenate([self.alpha_values, other.alpha_values]) if self.alpha_values is not None else other.alpha_values

@dataclass
class IntervalSplit:
    interpolation: tuple
    extrapolation_left: tuple
    extrapolation_right: tuple

def prepare_adv_diff_dataset(alpha_range=(0.01, 10), tau_range=(150, 400), dt=dt, n_samples=500):
    X = []
    X_tau = []
    t_values = []
    tau_values = []
    alpha_values = []
    TRANGE = (0.01, 2.0)
    while len(X) < n_samples:
        # sample alpha uniformly
        alpha = np.random.uniform(*alpha_range)
        t = np.random.uniform(*TRANGE)
        x_t = exact_solution(alpha, t)
        tau = np.random.randint(*tau_range)
        x_tau = exact_solution(alpha, t+(tau*dt))
        
        X.append(x_t)
        X_tau.append(x_tau)
        t_values.append(t)
        tau_values.append(tau)
        alpha_values.append(alpha)

    X = np.array(X)
    X_tau = np.array(X_tau)
    t_values = np.array(t_values)
    tau_values = np.array(tau_values)
    alpha_values = np.array(alpha_values)
    dataset = AdvectionDiffussionDataset(X, X_tau, t_values, tau_values, alpha_values)
    return dataset
   

def train_test_split_range(interval, interpolation_span=0.1, extrapolation_left_span=0.1, extrapolation_right_span=0.1):
    """
    Split the range into train and test ranges
    We have three test folds:
    1. Interpolation fold: Re and tau values are within the training (min, max) range but not in the training set
        We sample an interval of length x_interpolation_span% randomly from the total range
    2. Extrapolation fold: Re and tau values are outside the training (min, max) range
        We sample two intervals of length x_extrapolation_right_span% and x_extrapolation_left_span% from the total range
    3. Validation fold: Re and tau values are randomly sampled from the total set

    Overall interval looks like:
    Extrapolation_left_test | normal | Interpolation_test | normal | Extrapolation_right_test
    (min, extrapolation_left) | (extraplation_left, interpolation_min) | (interpolation_min, interpolation_max) | (interpolation_max, extrapolation_right) | (extrapolation_right, max)
    and
    train, val = split(normal, val_split)
    """
    r_min, r_max = interval
    length = (r_max-r_min)
    extra_left_length = extrapolation_left_span * length
    extra_right_length = extrapolation_right_span * length
    inter_length = interpolation_span * length

    extrapolation_left = (r_min, r_min + extra_left_length)
    extrapolation_right = (r_max - extra_right_length, r_max)

    interpolation_min = np.random.uniform(extrapolation_left[1], extrapolation_right[0] - inter_length)
    interpolation = (interpolation_min, interpolation_min + inter_length)

    train_ranges = [(extrapolation_left[1], interpolation[0]), (interpolation[1], extrapolation_right[0])]
    return IntervalSplit(interpolation, extrapolation_left, extrapolation_right), train_ranges

def get_train_ranges(interval_split):
    return [
        (interval_split.extrapolation_left[1], interval_split.interpolation[0]),
        (interval_split.interpolation[1], interval_split.extrapolation_right[0])
    ]

def get_train_val_test_folds(alpha_range, tau_range,
                             alpha_interpolation_span=0.10,
                             alpha_extrapolation_left_span=0.10,
                             alpha_extrapolation_right_span=0.10,
                             tau_interpolation_span=0.10,
                             tau_extrapolation_left_span=0.10,
                             tau_extrapolation_right_span=0.10,
                             n_samples_train=500,
                             n_samples_val=200):
    """
    Generate train (4 sub-regions) and val (left extrp, interp, right extrp
    for alpha x left extrp, interp, right extrp for tau) datasets.
    
    Returns:
        dataset_train  : AdvectionDiffussionDataset
        dataset_val    : AdvectionDiffussionDataset
        alpha_interval_split: IntervalSplit
        tau_interval_split  : IntervalSplit
    """

    # ---------------------------------------------------------------------
    # 1) Split alpha into 4 regions: left extrp, interp, right extrp, train
    # 2) Split tau   into 4 regions: left extrp, interp, right extrp, train
    # ---------------------------------------------------------------------
    alpha_interval_split, alpha_train_ranges = train_test_split_range(
        alpha_range,
        alpha_interpolation_span,
        alpha_extrapolation_left_span,
        alpha_extrapolation_right_span
    )
    tau_interval_split, tau_train_ranges = train_test_split_range(
        tau_range,
        tau_interpolation_span,
        tau_extrapolation_left_span,
        tau_extrapolation_right_span
    )

    # alpha_train_ranges and tau_train_ranges each have 2 intervals:
    #   alpha_train_ranges = [ (a1_lo, a1_hi), (a2_lo, a2_hi) ]
    #   tau_train_ranges   = [ (t1_lo, t1_hi), (t2_lo, t2_hi) ]
    #
    # Meanwhile, alpha_interval_split has:
    #   alpha_interval_split.extrapolation_left  = (a_left_lo, a_left_hi)
    #   alpha_interval_split.interpolation       = (a_int_lo, a_int_hi)
    #   alpha_interval_split.extrapolation_right = (a_right_lo, a_right_hi)
    # and similarly for tau_interval_split.

    # -------------------------------------------------------------
    # 3) Build the TRAIN dataset from the Cartesian product
    #    of alpha_train_ranges x tau_train_ranges => 4 combos
    # -------------------------------------------------------------
    dataset_train = AdvectionDiffussionDataset()
    for alpha_subrange in alpha_train_ranges:  # 2 intervals
        for tau_subrange in tau_train_ranges:  # 2 intervals
            subset = prepare_adv_diff_dataset(
                alpha_range=alpha_subrange,
                tau_range=tau_subrange,
                n_samples=n_samples_train
            )
            dataset_train.append(subset)

    # -------------------------------------------------------------
    # 4) Build the VAL dataset from the leftover intervals:
    #    alpha in { left extrp, interp, right extrp }
    #  x tau   in { left extrp, interp, right extrp } => up to 9 combos
    # -------------------------------------------------------------
    alpha_val_intervals = [
        alpha_interval_split.extrapolation_left,
        alpha_interval_split.interpolation,
        alpha_interval_split.extrapolation_right
    ]
    tau_val_intervals = [
        tau_interval_split.extrapolation_left,
        tau_interval_split.interpolation,
        tau_interval_split.extrapolation_right
    ]

    dataset_val = AdvectionDiffussionDataset()

    for a_val_range in alpha_val_intervals:
        for t_val_range in tau_val_intervals:
            subset_val = prepare_adv_diff_dataset(
                alpha_range=a_val_range,
                tau_range=t_val_range,
                n_samples=n_samples_val
            )
            dataset_val.append(subset_val)

    return dataset_train, dataset_val, alpha_interval_split, tau_interval_split


def plot_sample(dataset, i):
    """
    Plot a sample pair from the dataset.
    """
    X = dataset.X
    X_tau = dataset.X_tau
    t_values = dataset.t_values
    tau_values = dataset.tau_values
    alpha_values = dataset.alpha_values

    print("Shape of X:", X.shape)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    im1 = axs[0].imshow(X[i], extent=[0, 1, 0, 1], origin='lower', cmap='hot')
    axs[0].set_title(f'Initial State (t: {t_values[i]})')
    plt.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(X_tau[i], extent=[0, 1, 0, 1], origin='lower', cmap='hot')
    axs[1].set_title(f'Shifted State (t + tau): {t_values[i]+tau_values[i]*dt}')
    plt.colorbar(im2, ax=axs[1])

    fig.suptitle(f'Tau: {tau_values[i]}, Alpha: {alpha_values[i]:.4f}')
    plt.show()
    
    
def load_from_path(path):
    dataset_train_path = os.path.join(path, 'dataset_train.pkl')
    dataset_val_path = os.path.join(path, 'dataset_val.pkl')
    alpha_interval_split_path = os.path.join(path, 'alpha_interval_split.json')
    tau_interval_split_path = os.path.join(path, 'tau_interval_split.json')

    with open(dataset_train_path, 'rb') as f:
        dataset_train = pickle.load(f)
    with open(dataset_val_path, 'rb') as f:
        dataset_val = pickle.load(f)
    with open(alpha_interval_split_path, 'r') as f:
        alpha_interval_split = json.load(f)
        alpha_interval_split = IntervalSplit(**alpha_interval_split)
    with open(tau_interval_split_path, 'r') as f:
        tau_interval_split = json.load(f)
        tau_interval_split = IntervalSplit(**tau_interval_split)

    return dataset_train, dataset_val, alpha_interval_split, tau_interval_split
