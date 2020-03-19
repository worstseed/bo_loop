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
INIT_X_PRESENTATION = [2.5, 4, 6, 7, 8]
NUM_ACQ_OPTS = 10 # Number of times the acquisition function is optimized while looking for the next x to sample.

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

    logging.debug("Visualizing Look-Ahead with initial design {} and init {}".format(initial_design, init))
    # Initialize dummy dataset
    x, y = initialize_dataset(initial_design=initial_design, init=init)
    logging.debug("Initialized dataset with:\nsamples {0}\nObservations {1}".format(x, y))

    # Fit GP to the currently available dataset
    gp = GPR(kernel=Matern())
    logging.debug("Fitting GP to\nx: {}\ny:{}".format(x, y))
    gp.fit(x, y)  # fit the model
    mu_star_t_xy = get_mu_star(gp)
    logging.info("Mu-star at time t: {}".format(mu_star_t_xy))

    # noinspection PyStringFormat
    logging.debug("Model fit to dataset.\nOriginal Inputs: {0}\nOriginal Observations: {1}\n"
                  "Predicted Means: {2}\nPredicted STDs: {3}".format(x, y, *(gp.predict(x, return_std=True))))

    # Assume next evaluation location
    # x_ = np.mean(x, keepdims=True)
    #x_ = np.array([[5.8]])
    x_ = np.array([[5.0]])
    print(x_)
    y_ = f(x_[0])

    # Update dataset with new observation
    X2_ = np.append(x, x_, axis=0)
    Y2_ = y + [y_]

    logging.info("x: {}, y: {}".format(x_, y_))

    # Fit GP to the updated dataset
    gp2 = GPR(kernel=Matern())
    logging.debug("Fitting GP to\nx: {}\ny:{}".format(X2_, Y2_))
    gp2.fit(X2_, Y2_)  # fit the model
    mu_star_t1_xy = get_mu_star(gp2)
    logging.info("Mu-star at time t+1: {}".format(mu_star_t1_xy))

    # -------------------------Plotting madness begins---------------------------
    # Draw Figure 1.

    fig, ax = plt.subplots(1, 1, squeeze=True)
    fig.tight_layout()
    labels['gp_mean'] = r'Mean - $\mu^t(\cdot)$'
    # labels['incumbent'] = r'Incumbent - ${(\mu^*)}^t$'
    def draw_figure_1(ax):
        ax.set_xlim(xbounds)
        ax.set_ylim(ybounds)
        ax.grid()
        boplot.plot_objective_function(ax=ax)
        boplot.plot_gp(model=gp, confidence_intervals=[1.0, 2.0], ax=ax, custom_x=x)
        boplot.mark_observations(X_=x, Y_=y, mark_incumbent=False, ax=ax)

        ax.legend()
        ax.set_xlabel(labels['xlabel'])
        ax.set_ylabel(labels['gp_ylabel'])
        ax.set_title(r"Visualization of $\mathcal{G}^t$", loc='left')

    draw_figure_1(ax)
    boplot.highlight_configuration(mu_star_t_xy[0], lloc='bottom', ax=ax)
    boplot.highlight_output(mu_star_t_xy[1], label='${(\mu^*)}^{t}$', lloc='right', ax=ax)
    plt.show(plt.gcf())

    # End of figure 1.
    # ---------------------------------------
    # Draw Figure 2.

    fig, ax = plt.subplots(1, 1, squeeze=True)
    fig.tight_layout()
    labels['gp_mean'] = r'Mean - $\mu^{t+1}(\cdot)|_\lambda$'
    # labels['incumbent'] = r'Incumbent - ${(\mu^*)}^{t+1}|_\lambda$'

    def draw_figure_2(ax):
        ax.set_xlim(xbounds)
        ax.set_ylim(ybounds)
        ax.grid()
        boplot.plot_objective_function(ax=ax)
        boplot.plot_gp(model=gp2, confidence_intervals=[1.0, 2.0], ax=ax, custom_x=X2_)
        boplot.mark_observations(X_=X2_, Y_=Y2_, highlight_datapoint=np.where(np.isclose(X2_, x_))[0],
                                 mark_incumbent=False,
                                 highlight_label=r"Hypothetical Observation $<\lambda, c(\lambda)>$", ax=ax)

        ax.legend()
        ax.set_xlabel(labels['xlabel'])
        ax.set_ylabel(labels['gp_ylabel'])
        ax.set_title(r"Visualization of $\mathcal{G}^{t}|_\lambda$", loc='left')

    draw_figure_2(ax)
    boplot.highlight_configuration(mu_star_t1_xy[0], lloc='bottom', ax=ax)
    boplot.highlight_output(mu_star_t1_xy[1], label='${(\mu^*)}^{t+1}|_\lambda$', lloc='right', ax=ax)
    plt.show(plt.gcf())

    # End of figure 2.
    # ---------------------------------------
    # Draw Figure 3 for KG
    fig, (ax1, ax2) = plt.subplots(1, 2, squeeze=True)
    fig.tight_layout()
    labels['gp_mean'] = r'Mean - $\mu^t(\cdot)$'
    draw_figure_1(ax1)
    boplot.highlight_output(mu_star_t_xy[1], label='${(\mu^*)}^{t}$', lloc='right', ax=ax1)
    ax1.get_legend().remove()
    labels['gp_mean'] = r'Mean - $\mu^{t+1}(\cdot)|_\lambda$'
    draw_figure_2(ax2)
    boplot.highlight_output(mu_star_t1_xy[1], label='${(\mu^*)}^{t+1}|_\lambda$', lloc='left', ax=ax2)
    ax2.get_legend().remove()
    plt.show(plt.gcf())



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


    #init_size = max(1, int(args.num_func_evals * args.fraction_init))

    main(
        init_size=args.init_db_size,
        initial_design=args.initial_design,
    )