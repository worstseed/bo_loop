import numpy as np
from bo_loop_acq_functions import *


# Dictionaries for the graphs' appearance
colors = dict({
    'observations': 'black',
    'current_incumbent': 'red',
    'highlighted_point': 'red',
    'gp_mean': '#0F028A',
    'gp_variance': 'lightblue',
    'gp_variance_edge': 'k',
    'acq_fun': 'black',
    'envelope_min_opacity': 0.3,
    'envelope_max_opacity': 0.8,
})

# Various parameters for plotting required by our own code
params = {
    'sample_precision': 100, # Number of points to sample while plotting in the unit open interval [0, 1)
}

labels = dict({
    PI: 'Probability of Improvement',
    LCB: 'Lower Confidence Bound',
    EI: 'Expected Improvement',
    'xlabel': '$\lambda$',
    'ylabel': 'c($\lambda$)'
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


def f(x):
    return x[0]/5 * np.sin(x[0]) + x[0]/7 * np.cos(2 * x[0])

# bounds for the search
bounds = dict({
    'lower': 2,
    'upper': 9
})
