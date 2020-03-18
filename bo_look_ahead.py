import warnings
warnings.filterwarnings('ignore')
import argparse
import logging
from functools import partial

import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt

from bo_loop_acq_functions import EI, LCB, PI
from bo_plot_utils import plot_search_graph, plot_acquisition_function, acquisition_functions
from bo_loop_obj_fun import f, bounds


SEED = None
INIT_X_PRESENTATION = [3, 4, 4.6, 4.8, 5, 9.4, 10, 12.7]
NUM_ACQ_OPTS = 10 # Number of times the acquisition function is optimized while looking for the next x to sample.

def initialize_dataset(initial_design, init=None):
    """
    Initialize some data to start fitting the GP on.
    :param initial_design: Method for initializing the GP, choice between 'uniform', 'random', and 'presentation'
    :param init: Number of datapoints to initialize with, if relevant
    :return:
    """

    # sample initial query points
    if initial_design == 'uniform':
        x = np.linspace(bounds['lower'], bounds['upper'], init).reshape(-1, 1).tolist()
    elif initial_design == 'random':
        x = np.random.uniform(bounds['lower'], bounds['upper'], init).reshape(-1, 1).tolist()
    elif initial_design == 'presentation':
        x = np.array(INIT_X_PRESENTATION).reshape(-1, 1).tolist()

    # get corresponding response values
    y = list(map(f, x))

    return x, y


def show_gp_for_dataset(x, y):
    gp = Pipeline([["standardize", MinMaxScaler(feature_range=(0, 1))],
                   ["GP",
                    GPR(kernel=Matern(nu=2.5), normalize_y=True, n_restarts_optimizer=10, random_state=SEED)]])
    gp.fit(x, y)  # fit the model

    ax = plot_search_graph(x, y, gp)
    ax.legend()
    ax.grid()
    plt.show(plt.gcf())


def run_bo(acquisition, max_iter, initial_design, acq_add, init=None):
    """
    BO
    :param acquisition: type of acquisition function to be used
    :param max_iter: max number of function calls
    :param seed: seed used to keep experiments reproducible
    :param initial_design: Method for initializing the GP, choice between 'uniform', 'random', and 'presentation'
    :param acq_add: additional parameteres for acquisition function (e.g. kappa for LCB)
    :param init: Number of datapoints to initialize GP with.
    :return: all evaluated points.
    """

    logging.debug("Running BO with Acquisition Function {0}, maximum iterations {1}, initial design {2}, "
                  "acq_add {3} and init {4}".format(acquisition, max_iter, initial_design, acq_add, init))
    x, y = initialize_dataset(initial_design=initial_design, init=init)
    logging.debug("Initialized dataset with:\nsamples {0}\nObservations {1}".format(x, y))

    for i in range(1, max_iter):  # BO loop
        logging.debug('Sample #%d' % (i))

        gp = Pipeline([
            ["standardize", MinMaxScaler(feature_range=(0, 1))],
            ["GP", GPR(kernel=Matern(nu=2.5), normalize_y=True, n_restarts_optimizer=10, random_state=SEED)]
        ])
        gp.fit(x, y)  # fit the model
        # noinspection PyStringFormat
        logging.debug("Model fit to dataset.\nOriginal Inputs: {0}\nOriginal Observations: {1}\n"
                      "Predicted Means: {2}\nPredicted STDs: {3}".format(x, y, *(gp.predict(x, return_std=True))))
        ax = plot_search_graph(x, y, gp)

        # Partially initialize the acquisition function to work with the fmin interface
        # (only the x parameter is not specified)
        acqui = partial(acquisition, model=gp, eta=min(y), add=acq_add)
        plot_acquisition_function(acquisition, min(y), gp, acq_add, ax=ax)

        # optimize acquisition function, repeat 10 times, use best result
        x_ = None
        y_ = 10000
        # Feel free to adjust the hyperparameters
        for j in range(NUM_ACQ_OPTS):
            opt_res = minimize(acqui, np.random.uniform(bounds['lower'], bounds['upper']),
                               bounds=[[bounds['lower'], bounds['upper']]],
                               options={'maxfun': 20, 'maxiter': 20}, method="L-BFGS-B")
            if opt_res.fun[0] < y_:
                x_ = opt_res.x
                y_ = opt_res.fun[0]

        x.append(x_)
        y.append(f(x_))

        print("After {0}. loop iteration".format(i))
        print("x: {0:.3E}, y: {1:.3E}".format(x_[0], y_))
        # plot_search_graph(x, list(map(lambda x:-1*x, y)), gp)

        ax.legend()
        ax.grid()
        plt.show(plt.gcf())

    return y



def main(num_evals, init_size, repetitions, initial_design, acq_add, acquisition):
    for i in range(repetitions):
        bo_res_1 = run_bo(max_iter=num_evals, init=init_size, initial_design=initial_design, acquisition=acquisition, acq_add=acq_add)



if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser('AutoMLLecture')

    cmdline_parser.add_argument('-n', '--num_func_evals',
                                default=10,
                                help='Number of function evaluations',
                                type=int)
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
    cmdline_parser.add_argument('-a', '--acquisition',
                                default='LCB',
                                choices=['LCB', 'EI', 'PI'],
                                help='acquisition function')
    cmdline_parser.add_argument('-s', '--seed',
                                default=15,
                                help='Which seed to use',
                                required=False,
                                type=int)
    cmdline_parser.add_argument('-r', '--repetitions',
                                default=1,
                                help='Number of repeations for the experiment',
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

    main(   num_evals=args.num_func_evals,
            # init_size=init_size,
            init_size=args.init_db_size,
            repetitions=args.repetitions,
            initial_design=args.initial_design,
            acquisition=acquisition_functions[args.acquisition],
            acq_add=1
            )
