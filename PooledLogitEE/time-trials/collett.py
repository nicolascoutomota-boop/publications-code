######################################################################################################################
# Code for time-trials with data from Collett
#   Results for the run times are provided as a comment at the end
#
# Paul Zivich (Last update: 2026/2/25)
######################################################################################################################

import warnings
import numpy as np
import pandas as pd
from time import time
from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_plogit
from delicatessen.utilities import spline, plogit_predict

from standard import PooledLogitGComputation

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    #########################################################################
    # Setup data
    d = pd.read_csv("../data/collett.dat", sep=r'\s+',
                    names=['patient', 'time', 'delta', 'treat', 'init', 'size'])
    d['novel'] = d['treat'] - 1
    d['intercept'] = 1

    d1 = d.copy()
    d1['novel'] = 1
    d0 = d.copy()
    d0['novel'] = 0

    a = np.asarray(d['novel'])
    t = np.asarray(d['time'])
    y = np.asarray(d['delta'])
    W = np.asarray(d[['init', 'size', ]])

    ######################################
    # Example 1a: Disjoint Indicator
    print("DISJOINT INDICATOR")

    event_times = [0, ] + list(np.unique(d.loc[d['delta'] == 1, 'time'])) + [59, ]
    event_times_a1 = list(np.unique(d.loc[(d['delta'] == 1) & (d['novel'] == 1), 'time']))
    event_times_p1 = [0, ] + event_times_a1 + [59, ]
    event_times_a0 = list(np.unique(d.loc[(d['delta'] == 1) & (d['novel'] == 0), 'time']))
    event_times_p0 = [0, ] + event_times_a0 + [59, ]
    params_rd = len(event_times)
    params_r1 = len(event_times_p1)
    params_r0 = len(event_times_p0)
    params_plr_a1 = len(event_times_a1)
    params_plr_a0 = len(event_times_a0)

    def psi_plogit_a1(theta):
        ee_plog = ee_plogit(theta, t=t, delta=y, X=W, unique_times=event_times_a1)
        ee_plog = ee_plog * (a == 1)[None, :]
        return ee_plog

    def psi_plogit_a0(theta):
        ee_plog = ee_plogit(theta, t=t, delta=y, X=W, unique_times=event_times_a0)
        ee_plog = ee_plog * (a == 0)[None, :]
        return ee_plog

    def psi_rd(theta):
        # Extracting parameters
        rds = theta[:params_rd]
        idPLR = params_rd + W.shape[1] + params_plr_a1
        beta1 = theta[params_rd: idPLR]
        beta0 = theta[idPLR:]

        # Nuisance models
        ee_plog1 = psi_plogit_a1(theta=beta1)
        ee_plog0 = psi_plogit_a0(theta=beta0)

        # Predictions to get risk differences
        risk1 = plogit_predict(theta=beta1, delta=y, t=t, X=W,
                               times_to_predict=event_times, measure='risk', unique_times=event_times_a1)
        risk0 = plogit_predict(theta=beta0, delta=y, t=t, X=W,
                               times_to_predict=event_times, measure='risk', unique_times=event_times_a0)
        ee_rd = (risk1 - risk0) - np.asarray(rds)[:, None]

        # Returning stacked estimating equations
        return np.vstack([ee_rd, ee_plog1, ee_plog0])

    print("EE implementation")
    run_times = []
    for i in range(5):
        start = time()
        inits = ([0., ]*params_rd
                 + [0., ]*W.shape[1] + [-4., ] + [0., ]*(params_plr_a1 - 1)
                 + [0., ]*W.shape[1] + [-4., ] + [0., ]*(params_plr_a0 - 1))
        estr = MEstimator(psi_rd, init=inits)
        estr.estimate()
        run_times.append(time() - start)

    print("RUNTIME:", np.median(run_times))
    print(run_times)

    # print("Standard -- 1 CPU")
    run_times = []
    for i in range(5):
        start = time()
        plgc = PooledLogitGComputation(data=d, exposure='novel', time='time', delta='delta', verbose=False)
        plgc.outcome_model(model='novel*(init + size + C(time))')
        results = plgc.estimate(n_cpus=1, bs_iterations=1000, bs_method='frw', seed=80921)
        run_times.append(time() - start)

    print(results[['RD', 'Var_RD', 'LCL_RD', 'UCL_RD']].tail(1))
    print("RUNTIME:", np.median(run_times))
    print(run_times)

    print("Standard -- 7 CPU")
    run_times = []
    for i in range(5):
        start = time()
        plgc = PooledLogitGComputation(data=d, exposure='novel', time='time', delta='delta', verbose=False)
        plgc.outcome_model(model='novel*(init + size + C(time))')
        results = plgc.estimate(n_cpus=7, bs_iterations=1000, bs_method='frw', seed=80921)
        run_times.append(time() - start)

    print(results[['RD', 'Var_RD', 'LCL_RD', 'UCL_RD']].tail(1))
    print("RUNTIME:", np.median(run_times))
    print(np.min(run_times), np.max(run_times))

    ######################################
    # Example 1b: Splines
    print("SPLINES")

    t_steps = np.asarray(range(1, 60))
    tp_intervals = [0, ] + list(range(1, 59, 1)) + [59, ]
    params_risk = len(tp_intervals)

    intercept = np.ones(t_steps.shape)[:, None]
    time_splines = spline(t_steps, knots=[10, 20, 30, 40],
                          power=2, restricted=True, normalized=False)
    s_matrix = np.concatenate([intercept, t_steps[:, None], time_splines], axis=1)

    def psi_plogit_spline_a1(theta):
        ee_plog = ee_plogit(theta=theta, t=t, delta=y, X=W, S=s_matrix)
        ee_plog = ee_plog * (a == 1)[None, :]
        return ee_plog

    def psi_plogit_spline_a0(theta):
        ee_plog = ee_plogit(theta=theta, t=t, delta=y, X=W, S=s_matrix)
        ee_plog = ee_plog * (a == 0)[None, :]
        return ee_plog

    def psi_rd(theta):
        # Extracting parameters
        risks = theta[:params_risk]
        idPLRM = params_risk + 7
        beta1 = theta[params_risk:idPLRM]
        beta0 = theta[idPLRM:]

        # Nuisance models
        ee_plog1 = psi_plogit_spline_a1(theta=beta1)
        ee_plog0 = psi_plogit_spline_a0(theta=beta0)

        # Predictions to get risk differences
        risk1 = plogit_predict(theta=beta1, t=t, delta=y, X=W, S=s_matrix,
                               times_to_predict=tp_intervals, measure='risk')
        risk0 = plogit_predict(theta=beta0, t=t, delta=y, X=W, S=s_matrix,
                               times_to_predict=tp_intervals, measure='risk')
        ee_rd = (risk1 - risk0) - np.asarray(risks)[:, None]

        # Returning stacked estimating equations
        return np.vstack([ee_rd, ee_plog1, ee_plog0])

    print("EE implementation")
    run_times = []
    for i in range(5):
        start = time()
        inits = [0., ] * params_risk + [0., 0., -4., ] + [0., ]*4 + [0., 0., -4., ] + [0., ]*4
        estr = MEstimator(psi_rd, init=inits)
        estr.estimate()
        run_times.append(time() - start)

    print("RUNTIME:", np.mean(run_times))
    print(run_times)

    print("Standard -- 1 CPU")
    run_times = []
    for i in range(5):
        start = time()
        plgc = PooledLogitGComputation(data=d, exposure='novel', time='time', delta='delta', verbose=False)
        plgc.create_time_splines(term=2, knots=[10, 20, 30, 40])
        plgc.outcome_model(model='novel*(init + size + time + time_spline1 + time_spline2 + time_spline3)')
        results = plgc.estimate(n_cpus=1, bs_iterations=1000, bs_method='frw', seed=80921)
        run_times.append(time() - start)

    print(results[['RD', 'Var_RD', 'LCL_RD', 'UCL_RD']].tail(1))
    print("RUNTIME:", np.median(run_times))
    print(run_times)

    print("Standard -- 7 CPU")
    run_times = []
    for i in range(5):
        start = time()
        plgc = PooledLogitGComputation(data=d, exposure='novel', time='time', delta='delta', verbose=False)
        plgc.create_time_splines(term=2, knots=[10, 20, 30, 40])
        plgc.outcome_model(model='novel*(init + size + time + time_spline1 + time_spline2 + time_spline3)')
        results = plgc.estimate(n_cpus=7, bs_iterations=1000, bs_method='frw', seed=80921)
        run_times.append(time() - start)

    print(results[['RD', 'Var_RD', 'LCL_RD', 'UCL_RD']].tail(1))
    print("RUNTIME:", np.median(run_times))
    print(run_times)


# DISJOINT INDICATOR
#
# EE implementation
# RUNTIME: 0.19350552558898926
# [0.19215178489685059, 0.19582605361938477, 0.19156789779663086, 0.19350552558898926, 0.19875168800354004]
#             RD    Var_RD    LCL_RD    UCL_RD
# time
# 59   -0.189233  0.013719 -0.418799  0.040334
# RUNTIME: 708.8876039981842
# [580.8338131904602, 709.38179063797, 719.883208990097, 708.8876039981842, 566.2731094360352]
# Standard -- 7 CPU
#             RD    Var_RD    LCL_RD    UCL_RD
# time
# 59   -0.189233  0.013719 -0.418799  0.040334
# RUNTIME: 105.97134113311768
# 104.51006627082825 106.68051552772522
#
# SPLINES
#
# EE implementation
# RUNTIME: 0.7068254947662354
# [0.7460260391235352, 0.7252283096313477, 0.6905961036682129, 0.6862847805023193, 0.6859922409057617]
# Standard -- 1 CPU
#             RD    Var_RD    LCL_RD    UCL_RD
# time
# 59   -0.177775  0.015182 -0.419274  0.063724
# RUNTIME: 22.928261756896973
# [22.94381809234619, 22.928261756896973, 22.950029611587524, 22.85137915611267, 22.635519981384277]
# Standard -- 7 CPU
#             RD    Var_RD    LCL_RD    UCL_RD
# time
# 59   -0.177775  0.015182 -0.419274  0.063724
# RUNTIME: 6.67903995513916
# [6.684532403945923, 6.67903995513916, 6.605347156524658, 6.664432764053345, 6.8646087646484375]
