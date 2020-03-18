from matplotlib import pyplot as plt
import numpy as np
import logging

from bo_loop_acq_functions import LCB, EI, PI
from bo_configurations import *
from matplotlib import rcParams
# from datetime import datetime

rcParams["font.size"] = "16"
rcParams["axes.linewidth"] = 3
rcParams["lines.markersize"] = 15
rcParams["legend.loc"] = "lower right"
rcParams['axes.labelsize'] = 20


def get_plot_domain():
    """
    Generates the default domain of configuration values to be plotted.
    :return: A NumPy-array of shape [-1, 1]
    """

    return np.arange(xbounds[0], xbounds[1], 1 / params['sample_precision']).reshape(-1, 1)


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
    X_ = get_plot_domain()
    ax.plot(X_, f([X_]), linestyle='--', label="Objective function")

    return ax if return_flag else None


def mark_current_incumbent(x, y, invert_y=False, ax=None):
    """
    Convenience function to mark the current incumbent on the graph.
    :param x: Current incumbent's configuration.
    :param y: Current incumbent's observed cost.
    :param invert_y: Use the negative of the given y value, useful when switching between minimization and maximization.
    :param ax: A matplotlib.Axes.axes object on which the graphs are plotted. If None (default), a new 1x1 subplot is
    generated and the corresponding axes object is returned.
    :return: If ax is None, the matplotlib.Axes.axes object on which plotting took place, else None.
    """

    if invert_y:
        y = -y
    ax.scatter(x, y, color=colors['current_incumbent'], marker='v', label="Current incumbent", zorder=10)


def mark_observations(X_, Y_, mark_incumbent=True, ax=None):
    """
    Plots the given dataset as data observed thus far, including the current incumbent unless otherwise specified.
    :param X_: Configurations.
    :param Y_: Observed Costs.
    :param mark_incumbent: When True (default), distinctly marks the location of the current incumbent.
    :param ax: matplotlib.Axes.axes object given by the user, or newly generated for a 1x1 figure if None (default).
    :return: None if ax was given, otherwise the new matplotlib.Axes.axes object.
    """
    return_flag = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, squeeze=True)
        return_flag = True

    X_ = np.array(X_).reshape(-1, 1)
    Y_ = np.array(Y_).reshape(-1, 1)
    mask = np.ones(X_.shape[0], dtype=bool)
    logging.debug("Marking dataset with X of shape {} and Y of shape {}".format(X_.shape, Y_.shape))
    if mark_incumbent:
        incumb_idx = np.argmin(Y_)
        mark_current_incumbent(X_[incumb_idx, 0], Y_[incumb_idx, 0], ax=ax)
        mask[incumb_idx] = 0
    ax.scatter(X_[mask, 0], Y_[mask, 0], color=colors['observations'], marker='X', label="Observations", zorder=10)

    return ax if return_flag else None


def plot_gp(model, confidence_intervals=None, custom_x=None, ax=None):
    """
    Plot a GP's mean and, if required, its confidence intervals.
    :param model: GP
    :param confidence_intervals: If None (default) no confidence envelope is plotted. If a list of positive values
    [k1, k2, ...]is given, the confidence intervals k1*sigma, k2*sigma, ... are plotted.
    :param custom_x: (Optional) Numpy-array compatible list of x values that must be included in the plot.
    :param ax: A matplotlib.Axes.axes object on which the graphs are plotted. If None (default), a new 1x1 subplot is
    generated and the corresponding axes object is returned.
    :return: If ax is None, the matplotlib.Axes.axes object on which plotting took place, else None.
    """
    return_flag = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, squeeze=True)
        return_flag = True

    X_ = get_plot_domain()
    logging.debug("Generated x values for plotting of shape {0}".format(X_.shape))
    if custom_x is not None:
        custom_x = np.array(custom_x).reshape(-1, 1)
        logging.debug("Custom x has shape {0}".format(custom_x.shape))
        X_ = np.unique(np.vstack((X_, custom_x))).reshape(-1, 1)

    logging.debug("Plotting values for x of shape {0}".format(X_.shape))
    mu, sigma = model.predict(X_, return_std=True)
    logging.debug("Plotting GP with these values:\nSamples:\t\t{0}\nMeans:\t\t{1}\nSTDs:\t\t{2}".format(
        X_, mu, sigma
    ))

    # Plot the mean
    ax.plot(X_, mu, lw=2, color=colors['gp_mean'], label="GP Mean")

    # If needed, plot the confidence envelope(s)
    if confidence_intervals is not None:
        confidence_intervals = np.array(confidence_intervals)
        confidence_intervals.sort()

        # Dynamically generate opacities for each confidence envelope
        alphas = np.linspace(
            start=colors['envelope_max_opacity'],
            stop=colors['envelope_min_opacity'],
            num=confidence_intervals.shape[0],
            endpoint=False
        )

        for k, alpha in zip(confidence_intervals, alphas):
            ax.fill_between(
                X_[:, 0], mu - k*sigma, mu + k*sigma,
                facecolor=colors['gp_variance'], alpha=alpha,
                label="{0:.2f}-Sigma Confidence Envelope".format(k)
            )

    return ax if return_flag else None


# Plot objective function, observations (mark the newest) and the surrogate model (with mean and variance of GP)
def plot_complete_graph(X_, Y_, model, confidence_intervals=None, title=None, ax=None):
    """
    Plot a GP's mean, evaluated data-points, and its confidence intervals.
    :param X_: Configurations that have been evaluated
    :param Y_: Observed Costs
    :param model: GP
    :param confidence_intervals: If None (default) no confidence envelope is plotted. If a list of positive values
    [k1, k2, ...]is given, the confidence intervals k1*sigma, k2*sigma, ... are plotted.
    :param title: Title of the plot.
    :param ax: A matplotlib.Axes.axes object on which the graphs are plotted. If None (default), a new 1x1 subplot is
    generated and the corresponding axes object is returned.
    :return: If ax is None, the matplotlib.Axes.axes object on which plotting took place, else None.
    """
    return_flag = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, squeeze=True)
        return_flag = True

    # new_x = np.concatenate((np.linspace(bounds['lower'], bounds['upper'], 100), np.array(X_).reshape(-1)), axis=0)
    # new_x.sort()
    # mu, sigma = model.predict(new_x.reshape(-1, 1), return_std=True)
    # mu, sigma = -mu, -sigma

    plot_objective_function(ax=ax)
    plot_gp(model=model, confidence_intervals=confidence_intervals, ax=ax, custom_x=X_)

    # Mark Observations - Theoretically, there is no need to put this in a separate function and only plot_gp should be
    # used directly, but doing so allows better re-usability as well as avoids the need to import all the dictionaries
    # in other files.
    mark_observations(X_, Y_, mark_incumbent=True, ax=ax)

    # Disabled due to ambiguous interpretation of the utility of 'newest observation' - no need to put this in a function
    # ax.scatter(X_[-1], Y_[-1], color=colors['new_observation'], marker='v', label="Newest observation", zorder=10)
    ax.set_xlabel(labels['xlabel'])
    ax.set_ylabel(labels['ylabel'])
    ax.set_title(title)
    return ax if return_flag else None


# Plot acquisition function
def plot_acquisition_function(acquisition, eta, model, add=None, invert=False, ax=None):
    """
    Generate a plot to visualize the given acquisition function for the model.
    :param acquisition: Acquisition function handle, from bo_configurations.acquisition_functions.
    :param eta: Best observed value thus far.
    :param model: GP to be used as a model.
    :param add: Additional parameters passed to the acquisition function.
    :param invert: When True (default), it is assumed that the acquisition function needs to be inverted for plotting.
    :param ax: A matplotlib.Axes.axes object on which the graphs are plotted. If None (default), a new 1x1 subplot is
    generated and the corresponding axes object is returned.
    :return: If ax is None, the matplotlib.Axes.axes object on which plotting took place, else None.
    """
    return_flag = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, squeeze=True)
        return_flag = True

    X_ = get_plot_domain().reshape(-1)
    acquisition_fun = acquisition_functions[acquisition](X_, model=model, eta=eta, add=add)
    if invert:
        acquisition_fun = -acquisition_fun
    #zipped = list(zip(X_, acquisition_fun))
    #zipped.sort(key = lambda t: t[0])
    #X_, acquisition_fun = list(zip(*zipped))

    ax.plot(X_, acquisition_fun, color=colors['acq_fun'], label=labels[acquisition])
    ax.fill_between(X_, acquisition_fun, ybounds[0], facecolor=colors['acq_func_fill'])

    return ax if return_flag else None

    # now = datetime.now()
    # dt_string = now.strftime("%d_%m_%Y__%H_%M_%S.%f")
    # plt.savefig("../plots/bo_loop/" + dt_string + ".png")
    # plt.clf()


def indicate_next_sample(x, ybounds=ybounds, ax=None):
    """
    Draw a vertical line at the given configuration to indicate the next configuration to be sampled.
    :param x: Configuration.
    :param ybounds: A 2-tuple containing the upper and lower plotting bounds. By default uses bo_configurations.ybounds.
    :param ax: A matplotlib.Axes.axes object on which the graphs are plotted. If None (default), a new 1x1 subplot is
    generated and the corresponding axes object is returned.
    :return: If ax is None, the matplotlib.Axes.axes object on which plotting took place, else None.
    """
    return_flag = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, squeeze=True)
        return_flag = True

    ax.vlines(x, ymin=ybounds[0], ymax=ybounds[1], colors=colors['next_sample'], linestyles='dashed', label='Next Sample')
    ax.set_xticks(x, minor=True)
    xlabel = "{0:.2f}".format(x[0])
    ax.set_xticklabels([xlabel], {'color': colors['next_sample']}, minor=True)
    #logging.info("Xticks: {}".format(ax.xaxis.get_minor_ticks()))
    return ax if return_flag else None