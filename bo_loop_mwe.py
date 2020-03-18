import warnings
warnings.filterwarnings('ignore')
import argparse
import logging
from functools import partial

import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt

from bo_loop_acq_functions import EI, LCB, PI
from bo_plot_utils import plot_complete_graph, plot_acquisition_function, acquisition_functions
from bo_loop_obj_fun import f, bounds


def run_bo(acquisition, max_iter, init, seed, initial_design, acq_add):
    """
    BO
    :param acquisition: type of acquisition function to be used
    :param max_iter: max number of function calls
    :param init: number of points to build initial model
    :param seed: seed used to keep experiments reproducible
    :param random: if False initial points are linearly sampled in the bounds, otherwise uniformly random
    :param acq_add: additional parameteres for acquisition function (e.g. kappa for LCB)
    :return: all evaluated points.
    """
    # sample initial query points
    np.random.seed(seed)
    if initial_design == 'uniform':
        x = np.linspace(bounds['lower'], bounds['upper'], init).reshape(-1, 1).tolist()
    elif initial_design == 'random':
        x = np.random.uniform(bounds['lower'], bounds['upper'], init).reshape(-1, 1).tolist()
    elif initial_design == 'presentation':
        x = np.array([2, 3, 7, 4.6, 9.4, 10, 12.3]).reshape(-1, 1).tolist()

    # get corresponding response values
    y = list(map(f, x))


    for i in range(max_iter - init):  # BO loop
        logging.debug('Sample #%d' % (init + i))
        #Feel free to adjust the hyperparameters
        gp = GPR(kernel=Matern())
        gp.fit(x, y)  # fit the model

        # Partially initialize the acquisition function to work with the fmin interface
        # (only the x parameter is not specified)
        acqui = partial(acquisition, model=gp, eta=min(y), add=acq_add)

        # optimize acquisition function, repeat 10 times, use best result
        x_ = None
        y_ = 10000
        # Feel free to adjust the hyperparameters
        for j in range(10):
            opt_res = minimize(acqui, np.random.uniform(bounds['lower'], bounds['upper']),
                               bounds=[[bounds['lower'], bounds['upper']]], options={'maxfun': 20, 'maxiter': 20}, method="L-BFGS-B")
            if opt_res.fun[0] < y_:
                x_ = opt_res.x
                y_ = opt_res.fun[0]

        x.append(x_)
        y.append(f(x_))

        logging.info("After {0}. loop iteration".format(i))
        logging.info("x: {0:.3E}, y: {1:.3E}".format(x_[0], y_))
        # plot_search_graph(x, list(map(lambda x:-1*x, y)), gp)

        fig, ax = plt.subplots(1, 1, squeeze=True)
        plot_complete_graph(X_=x, Y_=y, model=gp, confidence_intervals=[1., 2., 3.], title='Search Graph', ax=ax)

        _ = plot_acquisition_function(acquisition, min(y), gp, acq_add, ax=ax)

        ax.legend()
        ax.grid()
        plt.show(plt.gcf())

    return y



def main(num_evals, init_size, repetitions, initial_design, seed, acq_add, acquisition):
    for i in range(repetitions):
        bo_res_1 = run_bo(max_iter=num_evals, init=init_size, initial_design=initial_design, acquisition=acquisition, acq_add=acq_add, seed=seed+1)

if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser('AutoMLLecture')

    cmdline_parser.add_argument('-n', '--num_func_evals',
                                default=15,
                                help='Number of function evaluations',
                                type=int)
    cmdline_parser.add_argument('-f', '--fraction_init',
                                default=0.3,
                                help='Fraction of budget (num_func_evals) to spend on building initial model',
                                type=float)
    cmdline_parser.add_argument('-i', '--initial_design',
                                default="uniform",
                                choices=['random', 'uniform', 'presentation'],
                                help='How to choose first observations.')
    cmdline_parser.add_argument('-v', '--verbose',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='verbosity')
    cmdline_parser.add_argument('-a', '--acquisition',
                                default='LCB',
                                choices=['LCB', 'EI', 'PI'],
                                help='acquisition function')
    cmdline_parser.add_argument('-s', '--seed',
                                default=14,
                                help='Which seed to use',
                                required=False,
                                type=int)
    cmdline_parser.add_argument('-r', '--repetitions',
                                default=1,
                                help='Number of repeations for the experiment',
                                required=False,
                                type=int)
    args, unknowns = cmdline_parser.parse_known_args()
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')

    init_size = max(1, int(args.num_func_evals * args.fraction_init))
    main(   num_evals=args.num_func_evals,
            init_size=init_size,
            repetitions=args.repetitions,
            initial_design=args.initial_design,
            acquisition=acquisition_functions[args.acquisition],
            seed=args.seed,
            acq_add=1
            )
