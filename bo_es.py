import warnings
warnings.filterwarnings('ignore')
import argparse
import logging
from functools import partial

import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern

from matplotlib import pyplot as plt

import bo_plot_utils as boplot
from bo_configurations import *


SEED = None
TOGGLE_PRINT = False
INIT_X_PRESENTATION = [2.5, 4, 6, 7, 8]

labels["xlabel"] = "$\lambda'$"
labels["ylabel"] = "$c(\lambda')$"


def initialize_dataset(initial_design, init=None):
    """
    Initialize some data to start fitting the GP on.
    :param initial_design: Method for initializing the GP, choice between 'uniform', 'random', and 'presentation'
    :param init: Number of datapoints to initialize with, if relevant
    :return:
    """

    # sample initial query points
    if initial_design == 'uniform':
        x = np.linspace(xbounds[0], xbounds[1], init).reshape(-1, 1).tolist()
    elif initial_design == 'random':
        x = np.random.uniform(xbounds[0], xbounds[1], init).reshape(-1, 1).tolist()
    elif initial_design == 'presentation':
        x = np.array(INIT_X_PRESENTATION).reshape(-1, 1).tolist()

    # get corresponding response values
    y = list(map(f, x))

    return x, y


def get_mu_star(model):
    """
    Given a model, return the (x, y) coords of mu-star.
    :param model: The underlying GP.
    :return:
    """

    X_ = boplot.get_plot_domain()
    mu = model.predict(X_).reshape(-1, 1)

    coords = np.hstack((X_, mu))
    idx = np.argmin(coords, axis=0)[1]
    return coords[idx, :]


def visualize_look_ahead(initial_design, init=None):
    """
    Visualize one-step of look-ahead.
    :param initial_design: Method for initializing the GP, choice between 'uniform', 'random', and 'presentation'
    :param init: Number of datapoints to initialize GP with.
    :return: None
    """

    boplot.set_rcparams(**{'figure.figsize': (21, 9)})

    logging.debug("Visualizing Look-Ahead with initial design {} and init {}".format(initial_design, init))
    # Initialize dummy dataset
    x, y = initialize_dataset(initial_design=initial_design, init=init)
    logging.debug("Initialized dataset with:\nsamples {0}\nObservations {1}".format(x, y))

    # Fit GP to the currently available dataset
    gp = GPR(kernel=Matern())
    logging.debug("Fitting GP to\nx: {}\ny:{}".format(x, y))
    gp.fit(x, y)  # fit the model

    histogram_precision = 20
    X_ = boplot.get_plot_domain(precision=histogram_precision)

    # 1. Plot GP and uniform prob of lambda*
    # 2. Sample GP, mark minima, update histogram of lambda*
    # 3. Repeat proces in 2.
    # 4. Show results after multiple iterations
    # 5. Mark the next sample based on MLE of posterior

    fig, (ax1, ax2) = plt.subplots(2, 1, squeeze=True)
    ax1.set_xlim(xbounds)
    ax1.set_ylim(gp_ybounds)
    ax2.set_xlim(xbounds)
    ax2.set_ylim((0, 1.0))
    ax1.grid()
    ax2.grid()

    boplot.plot_objective_function(ax=ax1)
    boplot.plot_gp(model=gp, confidence_intervals=[1.0], ax=ax1, custom_x=x)
    boplot.mark_observations(X_=x, Y_=y, mark_incumbent=False, ax=ax1)

    data_h1 = np.copy(X_)
    nbins = X_.shape[0]+1
    bin_range = (xbounds[0] - 1 / histogram_precision, xbounds[1] + 1 / histogram_precision)

    ax2.hist(
        data_h1, bins=nbins,
        range=bin_range, density=True,
        color='lightgreen', edgecolor='black'
    )

    ax1.legend()
    ax1.set_xlabel(labels['xlabel'])
    ax1.set_ylabel(labels['gp_ylabel'])
    ax1.set_title(r"Visualization of $\mathcal{G}^t$", loc='left')

    ax2.set_xlabel(labels['xlabel'])
    ax2.set_ylabel(r'$p_{min}$')
    ax2.set_title(r'Likelihood $\mathcal{L}(\lambda=\lambda^*)$', loc='left')

    plt.tight_layout()
    if TOGGLE_PRINT:
        plt.savefig('es_1')
    else:
        plt.show()

    # Draw one sample from gp

    fig, (ax1, ax2) = plt.subplots(2, 1, squeeze=True)
    ax1.set_xlim(xbounds)
    ax1.set_ylim(gp_ybounds)
    ax2.set_xlim(xbounds)
    ax2.set_ylim((0, 1.0))
    ax1.grid()
    ax2.grid()

    boplot.plot_objective_function(ax=ax1)
    boplot.mark_observations(X_=x, Y_=y, mark_incumbent=False, ax=ax1)

    nsamples = 1
    X_ = boplot.get_plot_domain(precision=histogram_precision)
    mu = gp.sample_y(X=X_, n_samples=nsamples, random_state=SEED)
    boplot.plot_gp_samples(
        mu=mu,
        nsamples=nsamples,
        precision=histogram_precision,
        custom_x=X_,
        show_min=True,
        ax=ax1
    )
    min_idx = np.argmin(mu, axis=0)
    data_h2 = np.append(data_h1, [X_[min_idx, 0]], axis=0)
    logging.info("Shape of data_h1 is {} and of data_h2 is {}".format(data_h1.shape, data_h2.shape))

    ax2.hist(
        data_h2, bins=nbins,
        range=bin_range, density=True,
        color='lightgreen', edgecolor='black'
    )

    ax1.legend()
    ax1.set_xlabel(labels['xlabel'])
    ax1.set_ylabel(labels['gp_ylabel'])
    ax1.set_title(r"One sample from $\mathcal{G}^t$", loc='left')

    ax2.set_xlabel(labels['xlabel'])
    ax2.set_ylabel(r'$p_{min}$')
    ax2.set_title(r'Likelihood $\mathcal{L}(\lambda=\lambda^*)$', loc='left')

    plt.tight_layout()
    if TOGGLE_PRINT:
        plt.savefig('es_2')
    else:
        plt.show()

    # Draw 3 samples from GP
    fig, (ax1, ax2) = plt.subplots(2, 1, squeeze=True)
    ax1.set_xlim(xbounds)
    ax1.set_ylim(gp_ybounds)
    ax2.set_xlim(xbounds)
    ax2.set_ylim((0, 1.0))
    ax1.grid()
    ax2.grid()

    boplot.plot_objective_function(ax=ax1)
    boplot.mark_observations(X_=x, Y_=y, mark_incumbent=False, ax=ax1)

    nsamples = 3
    X_ = boplot.get_plot_domain(precision=histogram_precision)
    mu = gp.sample_y(X=X_, n_samples=nsamples, random_state=SEED)
    boplot.plot_gp_samples(
        mu=mu,
        nsamples=nsamples,
        precision=histogram_precision,
        custom_x=X_,
        show_min=True,
        ax=ax1
    )
    min_idx = np.argmin(mu, axis=0)
    logging.info("Shape of X_ is {}".format(X_[min_idx, 0].shape))
    data_h2 = np.vstack((data_h1, X_[min_idx, 0].reshape(-1, 1)))
    logging.info("Shape of data_h1 is {} and of data_h2 is {}".format(data_h1.shape, data_h2.shape))

    ax2.hist(
        data_h2, bins=nbins,
        range=bin_range, density=True,
        color='lightgreen', edgecolor='black'
    )

    ax1.legend()
    ax1.set_xlabel(labels['xlabel'])
    ax1.set_ylabel(labels['gp_ylabel'])
    ax1.set_title(r"One sample from $\mathcal{G}^t$", loc='left')

    ax2.set_xlabel(labels['xlabel'])
    ax2.set_ylabel(r'$p_{min}$')
    ax2.set_title(r'Likelihood $\mathcal{L}(\lambda=\lambda^*)$', loc='left')

    plt.tight_layout()
    if TOGGLE_PRINT:
        plt.savefig('es_3')
    else:
        plt.show()

    # Draw 50 samples from GP

    fig, (ax1, ax2) = plt.subplots(2, 1, squeeze=True)
    ax1.set_xlim(xbounds)
    ax1.set_ylim(gp_ybounds)
    ax2.set_xlim(xbounds)
    ax2.set_ylim((0, 1.0))
    ax1.grid()
    ax2.grid()

    boplot.plot_objective_function(ax=ax1)
    boplot.mark_observations(X_=x, Y_=y, mark_incumbent=False, ax=ax1)

    nsamples = 50
    X_ = boplot.get_plot_domain(precision=histogram_precision)
    mu = gp.sample_y(X=X_, n_samples=nsamples, random_state=SEED)
    boplot.plot_gp_samples(
        mu=mu,
        nsamples=nsamples,
        precision=histogram_precision,
        custom_x=X_,
        show_min=False,
        ax=ax1
    )
    min_idx = np.argmin(mu, axis=0)
    logging.info("Shape of X_ is {}".format(X_[min_idx, 0].shape))
    data_h2 = np.vstack((data_h1, X_[min_idx, 0].reshape(-1, 1)))
    logging.info("Shape of data_h1 is {} and of data_h2 is {}".format(data_h1.shape, data_h2.shape))

    ax2.hist(
        data_h2, bins=nbins,
        range=bin_range, density=True,
        color='lightgreen', edgecolor='black'
    )

    ax1.legend()
    ax1.set_xlabel(labels['xlabel'])
    ax1.set_ylabel(labels['gp_ylabel'])
    ax1.set_title(r"One sample from $\mathcal{G}^t$", loc='left')
    ax1.legend().remove()

    ax2.set_xlabel(labels['xlabel'])
    ax2.set_ylabel(r'$p_{min}$')
    ax2.set_title(r'Likelihood $\mathcal{L}(\lambda=\lambda^*)$', loc='left')

    plt.tight_layout()
    if TOGGLE_PRINT:
        plt.savefig('es_4')
    else:
        plt.show()



def main(init_size, initial_design):
        visualize_look_ahead(
            init=init_size,
            initial_design=initial_design,
        )



if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser('AutoMLLecture')

    cmdline_parser.add_argument('-f', '--init_db_size',
                                default=4,
                                help='Size of the initial database',
                                type=int)
    cmdline_parser.add_argument('-i', '--initial_design',
                                default="random",
                                choices=['random', 'uniform', 'presentation'],
                                help='How to choose first observations.')
    cmdline_parser.add_argument('-v', '--verbose',
                                default=False,
                                help='verbosity',
                                action='store_true')
    cmdline_parser.add_argument('-s', '--seed',
                                default=15,
                                help='Which seed to use',
                                required=False,
                                type=int)
    cmdline_parser.add_argument('-p', '--print',
                                default=False,
                                help='Print graphs to file instead of displaying on screen.',
                                action='store_true')

    args, unknowns = cmdline_parser.parse_known_args()
    log_lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')

    # init_size = max(1, int(args.num_func_evals * args.fraction_init))
    # Seed the RNG to obtain reproducible results
    SEED = args.seed
    np.random.seed(SEED)

    TOGGLE_PRINT = args.print
    if TOGGLE_PRINT:
        boplot.enable_printing()
    else:
        boplot.enable_onscreen_display()

    #init_size = max(1, int(args.num_func_evals * args.fraction_init))

    main(
        init_size=args.init_db_size,
        initial_design=args.initial_design,
    )