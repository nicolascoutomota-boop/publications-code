######################################################################################################################
# Code to simulate the true values for the simulation experiments
#
# Paul Zivich (Last update: 2025/4/17)
######################################################################################################################

import numpy as np
from lifelines import KaplanMeierFitter

from dgm import dgm

np.random.seed(999878)

d = dgm(n=10000000, truth=True)

km1 = KaplanMeierFitter()
km1.fit(d['T1_star'], d['delta1'])
risk1 = 1-km1.survival_function_at_times([5, 10, 15, 20, 25, 30])
km0 = KaplanMeierFitter()
km0.fit(d['T0_star'], d['delta0'])
risk0 = 1-km0.survival_function_at_times([5, 10, 15, 20, 25, 30])
true_rd = risk1 - risk0
true_rd.to_csv("truth.csv")
