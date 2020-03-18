from matplotlib import pyplot as plt
import numpy as np
import logging

from bo_loop_acq_functions import LCB, EI, PI
from bo_loop_obj_fun import f, bounds
from matplotlib import rcParams
# from datetime import datetime

rcParams["font.size"] = "15"
rcParams["axes.linewidth"] = 3
rcParams["lines.markersize"] = 15
rcParams["legend.loc"] = "lower right"
rcParams['axes.labelsize'] = 20



# Dictionaries for the graphs' appearance
colors = dict({
    'observations': 'black',
    'new_observation': 'red',
    'highlighted_point': 'red',
    'gp_mean': '#0F028A',
    'gp_variance': 'lightblue',
    'gp_variance_edge': 'k',
    'acq_fun': 'black'
})

labels = dict({
    PI: 'Probability of Improvement',
    LCB: 'Lower Confidence Bound',
    EI: 'Expected Improvement',
})

ylabels = dict({
    PI: 'PI(x)',
    LCB: 'LCB(x)',
    EI: 'EI(x)'
})

acquisition_functions = dict({
    PI: PI,
    LCB: LCB,
    EI: EI,
    'PI': PI,
    'LCB': LCB,
    'EI': EI
})

# Plot objective function, defined f(x)
def plot_objective_function(ax=None):
    """
    Plots the underlying true objective function being used for BO.
    :param ax: matplotlib.Axes.axes object given by the user, or newly generated for a 1x1 figure if None (default).
    :return: None if ax was given, otherwise the new matplotlib.Axes.axes object.
    """
    return_flag = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, squeeze=True)
        return_flag = True
    axis = np.arange(start=bounds['lower'], stop=bounds['upper'], step=0.1)
    ax.plot(axis, f([axis]), linestyle='--', label="Objective function")

    return ax if return_flag else None

# Plot objective function, observations (mark the newest) and the surrogate model (with mean and variance of GP)
def plot_search_graph(observed_x, observed_y, model, ax=None):
    return_flag = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, squeeze=True)
        return_flag = True

    new_x = np.concatenate((np.linspace(bounds['lower'], bounds['upper'], 100), np.array(observed_x).reshape(-1)), axis=0)
    new_x.sort()
    mu, sigma = model.predict(new_x.reshape(-1, 1), return_std=True)
    logging.debug("Plotting GP with these values:\nSamples:\t\t{0}\nMeans:\t\t{1}\nSTDs:\t\t{2}".format(
        new_x, mu, sigma
    ))
    # mu, sigma = -mu, -sigma

    plot_objective_function(ax=ax)
    ax.plot(new_x, mu, lw=2, color=colors['gp_mean'], label="GP mean")
    ax.fill_between(new_x, mu+3*sigma, mu-3*sigma, facecolor=colors['gp_variance'], alpha=0.3, label="Sigma Confidence Envelope")
    ax.fill_between(new_x, mu+2*sigma, mu-2*sigma, facecolor=colors['gp_variance'], alpha=0.4)
    ax.fill_between(new_x, mu+sigma, mu-sigma, facecolor=colors['gp_variance'], alpha=0.5)
    ax.scatter(observed_x[:-1], observed_y[:-1], color=colors['observations'], marker='X', label="Observations (" + str(len(observed_x)-1) + ")", zorder=10)
    ax.scatter(observed_x[-1], observed_y[-1], color=colors['new_observation'], marker='v', label="Newest observation", zorder=10)
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("$c(\lambda)$")
    ax.set_title('Search graph')
    return ax if return_flag else None

# Plot acquisition function
def plot_acquisition_function(acquisition, eta, model, add=None, ax=None):
    return_flag = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, squeeze=True)
        return_flag = True

    new_x = np.linspace(bounds['lower'], bounds['upper'], 1000)
    acquisition_fun = acquisition_functions[acquisition](new_x, model=model, eta=eta, add=add, plotting=True)
    zipped = list(zip(new_x, acquisition_fun))
    zipped.sort(key = lambda t: t[0])
    new_x, acquisition_fun = list(zip(*zipped))

    ax.plot(new_x, acquisition_fun, color=colors['acq_fun'], label=labels[acquisition])
    # ax.set_xlabel("x")
    # ax.set_ylabel(ylabels[acquisition])

    return ax if return_flag else None

    # now = datetime.now()
    # dt_string = now.strftime("%d_%m_%Y__%H_%M_%S.%f")
    # plt.savefig("../plots/bo_loop/" + dt_string + ".png")
    # plt.clf()
