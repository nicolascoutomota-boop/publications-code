######################################################################################################################
# Code for time-trials with data from Lau et al.
#   Results for the run times are provided as a comment at the end
#
# Paul Zivich (Last update: 2025/4/17)
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
    d = pd.read_csv("../data/lau.csv")

    # Generating splines for continuous variables
    d[['cd4_sp1', 'cd4_sp2']] = spline(d['cd4nadir'], knots=[2.1, 3.5, 5.2], power=2, restricted=True, normalized=False)
    d[['age_sp1', 'age_sp2']] = spline(d['ageatfda'], knots=[25, 35, 50], power=2, restricted=True, normalized=False)

    # Maxing maximum follow-up 10 years
    d['event'] = np.where(d['eventtype'] == 2, 1, 0)
    d['event'] = np.where(d['t'] > 10, 0, d['event'])
    d['t'] = np.where(d['t'] > 10, 10, d['t'])

    # Transforming time into days
    d['days'] = np.ceil(d['t'] * 365.25)
    d['days'] = d['days'].astype(int)
    d['months'] = np.ceil(d['days'] / 30.437)
    d['months'] = d['months'].astype(int)

    time_var = 'days'
    delta_var = 'event'
    act_var = 'BASEIDU'
    a = np.asarray(d[act_var])
    t = np.asarray(d[time_var])
    y = np.asarray(d[delta_var])
    W = np.asarray(d[['black', 'cd4nadir', 'cd4_sp1', 'cd4_sp2', 'ageatfda', 'age_sp1', 'age_sp2']])

    ##########################################
    # Example 2a1: Disjoint Indicator - Days
    print("DISJOINT INDICATOR")

    event_times = [0, ] + list(np.unique(d.loc[d[delta_var] == 1, time_var])) + [np.max(t), ]
    event_times_a1 = list(np.unique(d.loc[(d[delta_var] == 1) & (d[act_var] == 1), time_var]))
    event_times_p1 = [0, ] + event_times_a1 + [np.max(t), ]
    event_times_a0 = list(np.unique(d.loc[(d[delta_var] == 1) & (d[act_var] == 0), time_var]))
    event_times_p0 = [0, ] + event_times_a0 + [np.max(t), ]
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
    # run_times = []
    # for i in range(5):
    #     start = time()
    #     inits = [0., ] * W.shape[1] + [-5., ] + [0., ] * (params_plr_a1 - 1)
    #     estr = MEstimator(psi_plogit_a1, init=inits)
    #     estr.estimate()
    #     starting_a1 = list(estr.theta)
    #     inits = [0., ] * W.shape[1] + [-5., ] + [0., ] * (params_plr_a0 - 1)
    #     estr = MEstimator(psi_plogit_a0, init=inits)
    #     estr.estimate()
    #     starting_a0 = list(estr.theta)
    #     inits = [0., ] * params_rd + starting_a1 + starting_a0
    #     estr = MEstimator(psi_rd, init=inits)
    #     estr.estimate()
    #     run_times.append(time() - start)

    # print("RUNTIME:", np.median(run_times))
    # print(run_times)

    # print("Standard -- 1 CPU")
    # start = time()
    # plgc = PooledLogitGComputation(data=d, exposure=act_var, time=time_var, delta=delta_var, verbose=False)
    # plgc.outcome_model(model='BASEIDU*(black + cd4nadir + cd4_sp1 + cd4_sp2 + ageatfda + age_sp1 + age_sp2 + C(days)')
    # plgc.estimate(n_cpus=1, bs_iterations=1000, seed=80921)
    # print(time() - start)

    # print("Standard -- 7 CPU")
    # MemoryError
    # start = time()
    # plgc = PooledLogitGComputation(data=d, exposure=act_var, time=time_var, delta=delta_var, verbose=False)
    # plgc.outcome_model(model='BASEIDU*(black + cd4nadir + cd4_sp1 + cd4_sp2 + ageatfda + age_sp1 + age_sp2 + C(days)')
    # plgc.estimate(n_cpus=7, bs_iterations=1000, seed=80921)
    # print(time() - start)

    ##########################################
    # Example 2b1: Splines - Days
    print("SPLINES")

    max_time = int(np.max(t))
    t_steps = np.asarray(range(1, max_time + 1))
    tp_intervals = [0, ] + list(range(1, max_time + 1, 50)) + [np.max(t), ]
    params_risk = len(tp_intervals)
    intercept = np.ones(t_steps.shape)[:, None]
    time_splines = spline(t_steps, knots=[500, 1000, 2000, 3000, 3500],
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

    def psi_rd_spline(theta):
        # Extracting parameters
        risks = theta[:params_risk]
        idPLRM = params_risk + W.shape[1] + s_matrix.shape[1]
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

    # print("EE implementation")
    # run_times = []
    # for i in range(5):
    #     start = time()
    #     inits = [0., ] * W.shape[1] + [-8., ] + [0., ] * 5
    #     estr = MEstimator(psi_plogit_spline_a1, init=inits)
    #     estr.estimate()
    #     starting_a1 = list(estr.theta)
    #     inits = [0., ] * W.shape[1] + [-8., ] + [0., ] * 5
    #     estr = MEstimator(psi_plogit_spline_a0, init=inits)
    #     estr.estimate()
    #     starting_a0 = list(estr.theta)
    #     inits = [0., ] * params_risk + starting_a1 + starting_a0
    #     estr = MEstimator(psi_rd_spline, init=inits)
    #     estr.estimate()
    #     run_times.append(time() - start)
    #
    # print("RUNTIME:", np.median(run_times))
    # print(run_times)

    # print("Standard -- 1 CPU")
    # start = time()
    # plgc = PooledLogitGComputation(data=d, exposure=act_var, time=time_var, delta=delta_var, verbose=False)
    # plgc.create_time_splines(term=2, knots=[500, 1000, 2000, 3000, 3500])
    # plgc.outcome_model(model='BASEIDU*(black + cd4nadir + cd4_sp1 + cd4_sp2 + ageatfda + age_sp1 + age_sp2 '
    #                          '+ days + days_spline1 + days_spline2 + days_spline3 + days_spline4)')
    # results = plgc.estimate(n_cpus=1, bs_iterations=1000, seed=80921)
    # print(results[['RD', 'Var_RD', 'LCL_RD', 'UCL_RD']].tail(1))
    # print(time() - start)

    # print("Standard -- 7 CPU")
    # MemoryError
    # start = time()
    # plgc = PooledLogitGComputation(data=d, exposure=act_var, time=time_var, delta=delta_var, verbose=False)
    # plgc.create_time_splines(term=2, knots=[500, 1000, 2000, 3000, 3500])
    # plgc.outcome_model(model='BASEIDU*(black + cd4nadir + cd4_sp1 + cd4_sp2 + ageatfda + age_sp1 + age_sp2 '
    #                          '+ days + days_spline1 + days_spline2 + days_spline3 + days_spline4)')
    # results = plgc.estimate(n_cpus=7, bs_iterations=1000, seed=80921)
    # print(results[['RD', 'Var_RD', 'LCL_RD', 'UCL_RD']].tail(1))
    # print(time() - start)

    ################################################################
    # Switching to Months
    time_var = 'months'
    t = np.asarray(d[time_var])

    event_times = [0, ] + list(np.unique(d.loc[d[delta_var] == 1, time_var])) + [np.max(t), ]
    event_times_a1 = list(np.unique(d.loc[(d[delta_var] == 1) & (d[act_var] == 1), time_var]))
    event_times_p1 = [0, ] + event_times_a1 + [np.max(t), ]
    event_times_a0 = list(np.unique(d.loc[(d[delta_var] == 1) & (d[act_var] == 0), time_var]))
    event_times_p0 = [0, ] + event_times_a0 + [np.max(t), ]
    params_rd = len(event_times)
    params_r1 = len(event_times_p1)
    params_r0 = len(event_times_p0)
    params_plr_a1 = len(event_times_a1)
    params_plr_a0 = len(event_times_a0)

    ##########################################
    # Example 2a1: Disjoint Indicator - Months
    print("MONTHS")
    print("DISJOINT INDICATOR")

    # print("EE implementation")
    # run_times = []
    # for i in range(5):
    #     start = time()
    #     inits = [0., ] * W.shape[1] + [-2., ] + [0., ] * (params_plr_a1 - 1)
    #     estr = MEstimator(psi_plogit_a1, init=inits)
    #     estr.estimate()
    #     starting_a1 = list(estr.theta)
    #     inits = [0., ] * W.shape[1] + [-2., ] + [0., ] * (params_plr_a0 - 1)
    #     estr = MEstimator(psi_plogit_a0, init=inits)
    #     estr.estimate()
    #     starting_a0 = list(estr.theta)
    #     inits = [0., ] * params_rd + starting_a1 + starting_a0
    #     estr = MEstimator(psi_rd, init=inits)
    #     estr.estimate()
    #     run_times.append(time() - start)

    # print("RUNTIME:", np.median(run_times))
    # print(run_times)

    # print("Standard -- 1 CPU")
    # start = time()
    # plgc = PooledLogitGComputation(data=d, exposure='BASEIDU', time='months', delta='event', verbose=False)
    # plgc.outcome_model(model='BASEIDU*(black + cd4nadir + cd4_sp1 + cd4_sp2 + ageatfda + age_sp1 + age_sp2'
    #                          '+ C(months))')
    # results = plgc.estimate(n_cpus=1, bs_iterations=1000, seed=80921)
    # print(results[['RD', 'Var_RD', 'LCL_RD', 'UCL_RD']].tail(1))
    # print(time() - start)
    # Only run once since takes hours (not much gained by repeating)

    # print("Standard -- 7 CPU")
    # start = time()
    # plgc = PooledLogitGComputation(data=d, exposure='BASEIDU', time='months', delta='event', verbose=False)
    # plgc.outcome_model(model='BASEIDU*(black + cd4nadir + cd4_sp1 + cd4_sp2 + ageatfda + age_sp1 + age_sp2'
    #                          '+ C(months))')
    # results = plgc.estimate(n_cpus=7, bs_iterations=1000, seed=80921)
    # print(results[['RD', 'Var_RD', 'LCL_RD', 'UCL_RD']].tail(1))
    # print(time() - start)
    # Only run once since takes hours (not much gained by repeating)

    ##########################################
    # Example 2a2: Splines - Months
    print("SPLINES")

    max_time = int(np.max(t))
    t_steps = np.asarray(range(1, max_time + 1))
    tp_intervals = [0, ] + list(range(1, max_time + 1, 5)) + [np.max(t), ]
    params_risk = len(tp_intervals)
    intercept = np.ones(t_steps.shape)[:, None]
    time_splines = spline(t_steps, knots=[16, 32, 66, 100, 117],
                          power=2, restricted=True, normalized=False)
    s_matrix = np.concatenate([intercept, t_steps[:, None], time_splines], axis=1)

    print("EE implementation")
    run_times = []
    for i in range(5):
        start = time()
        inits = [0., ]*W.shape[1] + [-2., ] + [0., ]*5
        estr = MEstimator(psi_plogit_spline_a1, init=inits)
        estr.estimate()
        starting_a1 = list(estr.theta)
        inits = [0., ]*W.shape[1] + [-2., ] + [0., ]*5
        estr = MEstimator(psi_plogit_spline_a0, init=inits)
        estr.estimate()
        starting_a0 = list(estr.theta)
        inits = [0., ]*params_risk + starting_a1 + starting_a0
        estr = MEstimator(psi_rd_spline, init=inits)
        estr.estimate()
        estr.print_results(subset=[params_risk-1, ], decimals=3)
        run_times.append(time() - start)

    print("RUNTIME:", np.median(run_times))
    print(run_times)

    print("Standard -- 1 CPU")
    run_times = []
    for i in range(5):
        start = time()
        plgc = PooledLogitGComputation(data=d, exposure='BASEIDU', time='months', delta='event', verbose=False)
        plgc.create_time_splines(term=2, knots=[16, 32, 66, 100, 117])
        plgc.outcome_model(model='BASEIDU*(black + cd4nadir + cd4_sp1 + cd4_sp2 + ageatfda + age_sp1 + age_sp2 '
                                 '+ months + months_spline1 + months_spline2 + months_spline3 + months_spline4)')
        results = plgc.estimate(n_cpus=1, bs_iterations=1000, seed=80921)
        run_times.append(time() - start)

    print(results[['RD', 'Var_RD', 'LCL_RD', 'UCL_RD']].tail(1))
    print("RUNTIME:", np.median(run_times))
    print(run_times)

    print("Standard -- 7 CPU")
    run_times = []
    for i in range(5):
        start = time()
        plgc = PooledLogitGComputation(data=d, exposure='BASEIDU', time='months', delta='event', verbose=False)
        plgc.create_time_splines(term=2, knots=[16, 32, 66, 100, 117])
        plgc.outcome_model(model='BASEIDU*(black + cd4nadir + cd4_sp1 + cd4_sp2 + ageatfda + age_sp1 + age_sp2 '
                                 '+ months + months_spline1 + months_spline2 + months_spline3 + months_spline4)')
        results = plgc.estimate(n_cpus=7, bs_iterations=1000, seed=80921)
        run_times.append(time() - start)

    print(results[['RD', 'Var_RD', 'LCL_RD', 'UCL_RD']].tail(1))
    print("RUNTIME:", np.median(run_times))
    print(run_times)


# --- DAYS ---
#
# DISJOINT INDICATOR
# EE implementation
# RUNTIME: 55.033812284469604
# [55.101975440979004, 55.033812284469604, 54.94883751869202, 55.08584952354431, 55.00955295562744]
#
# SPLINES
#
# EE implementation
# RUNTIME: 113.41403222084045
# [136.19984817504883, 137.38442587852478, 113.23208165168762, 113.41403222084045, 113.27399015426636]
#
# Standard -- 1 CPU
#             RD    Var_RD    LCL_RD    UCL_RD
# days
# 3653  0.160647  0.002356  0.065515  0.255778
# 15366.362752199173
#
# Standard -- 7 CPU
# MemoryError
#
# --- MONTHS ---
#
# DISJOINT INDICATOR
# EE implementation
# RUNTIME: 5.028807878494263
# [5.028807878494263, 5.076036214828491, 5.121047019958496, 4.951996326446533, 3.8277430534362793]
# Standard -- 1 CPU
#               RD    Var_RD    LCL_RD    UCL_RD
# months
# 121     0.162701  0.002397  0.066737  0.258665
# 9301.929284095764
#
#
# SPLINES
# EE implementation
# RUNTIME: 2.6674365997314453
# [2.6630606651306152, 2.6716809272766113, 2.686751365661621, 2.6393511295318604, 2.6674365997314453]
# Standard -- 1 CPU
#               RD    Var_RD    LCL_RD    UCL_RD
# months
# 121     0.161423  0.002364  0.066125  0.256721
# RUNTIME: 439.49853467941284
# [343.9984362125397, 378.1661558151245, 441.1363580226898, 439.49853467941284, 440.5179286003113]
# Standard -- 7 CPU
#               RD    Var_RD    LCL_RD    UCL_RD
# months
# 121     0.161423  0.002364  0.066125  0.256721
# RUNTIME: 171.4404480457306
# [148.55218052864075, 171.4404480457306, 191.4471776485443, 197.53114652633667, 153.27429294586182]

