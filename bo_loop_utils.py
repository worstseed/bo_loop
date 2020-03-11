from matplotlib import pyplot as plt
import numpy as np

from bo_loop_acq_functions import LCB, EI, PI
from bo_loop_obj_fun import f, bounds
# from datetime import datetime


# Dictionaries for the graphs' appearance
colors = dict({
    'observations': 'grey',
    'new_observation': 'black',
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
def plot_objective_function():
    axis = np.arange(start=bounds['lower'], stop=bounds['upper'], step=0.1)
    plt.plot(axis, f([axis]), linestyle='--', label="Objective function")
    plt.legend()
    plt.grid()

# Plot objective function, observations (mark the newest) and the surrogate model (with mean and variance of GP)
def plot_search_graph(observed_x, observed_y, model):
    plt.figure(1)

    new_x = np.linspace(bounds['lower'], bounds['upper'], 100)
    mu, sigma = model.predict(new_x.reshape(-1, 1), return_std=True)
    # mu, sigma = -mu, -sigma

    plot_objective_function()
    plt.plot(new_x, mu, lw=2, color=colors['gp_mean'], label="GP mean")
    plt.fill_between(new_x, mu+3*sigma, mu-3*sigma, facecolor=colors['gp_variance'], edgecolor=colors['gp_variance_edge'], alpha=0.3, label="GP 3*std")
    plt.fill_between(new_x, mu+2*sigma, mu-2*sigma, facecolor=colors['gp_variance'], edgecolor=colors['gp_variance_edge'], alpha=0.4, label="GP 2*std")
    plt.fill_between(new_x, mu+sigma, mu-sigma, facecolor=colors['gp_variance'], edgecolor=colors['gp_variance_edge'], alpha=0.5, label="GP std")
    plt.scatter(observed_x[:-1], observed_y[:-1], color=colors['observations'], marker='X', label="Observations (" + str(len(observed_x)-1) + ")")
    plt.scatter(observed_x[-1], observed_y[-1], color=colors['new_observation'], marker='v', label="Newest observation")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title('Search graph')

# Plot acquisition function
def plot_acquisition_function(acquisition, eta, model, add=None):
    new_x = np.linspace(bounds['lower'], bounds['upper'], 1000)
    acquisition_fun = acquisition_functions[acquisition](new_x, model=model, eta=eta, add=add, plotting=True)
    zipped = list(zip(new_x, acquisition_fun))
    zipped.sort(key = lambda t: t[0])
    new_x, acquisition_fun = list(zip(*zipped))

    plt.plot(new_x, acquisition_fun, color=colors['acq_fun'], label=labels[acquisition])
    plt.xlabel("x")
    plt.ylabel(ylabels[acquisition])
    plt.legend()
    plt.show()

    # now = datetime.now()
    # dt_string = now.strftime("%d_%m_%Y__%H_%M_%S.%f")
    # plt.savefig("../plots/bo_loop/" + dt_string + ".png")
    # plt.clf()
