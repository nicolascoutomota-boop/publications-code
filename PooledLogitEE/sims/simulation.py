######################################################################################################################
# Code to run the simulation experiments
#   Results are shown as comments at the end
#
# Paul Zivich (Last update: 2026/2/24)
######################################################################################################################

import numpy as np
import pandas as pd

from dgm import dgm
from estimators import PooledLogitEE

# Setup for simulation
runs = 5000
n_obs = 2000
np.random.seed(338092421 - 250 + n_obs)

# Loading the truth values (calculated separately)
truth = pd.read_csv("truth.csv")
truth = np.asarray(truth['KM_estimate'])
true_5, true_10, true_15, true_20, true_25, true_30 = truth

# Storage for results at each time
results_5, results_10, results_15, results_20, results_25, results_30 = [], [], [], [], [], []

for i in range(runs):
    print("...starting", i+1)
    result5_row, result10_row, result15_row, result20_row, result25_row, result30_row = [], [], [], [], [], []
    d = dgm(n=n_obs, truth=False)

    # Pooled Logit Specifications: Exponential, Gompertz, Weibull, splines, disjoint
    for funcform_time in ['constant', 'linear', 'log', 'spline', 'disjoint']:
        try:
            plee = PooledLogitEE(data=d, time='T_star', delta='delta', action='A')
            plee.nuisance_model(covariates=['W', 'W_sp1', 'W_sp2'], time=funcform_time)
            plee.estimate()
            for j in [plee.point, plee.variance, plee.lower_ci, plee.upper_ci]:
                result5_row.append(j[0])
                result10_row.append(j[1])
                result15_row.append(j[2])
                result20_row.append(j[3])
                result25_row.append(j[4])
                result30_row.append(j[5])
        except:
            for j in [1, 2, 3, 4]:
                result5_row.append(np.nan)
                result10_row.append(np.nan)
                result15_row.append(np.nan)
                result20_row.append(np.nan)
                result25_row.append(np.nan)
                result30_row.append(np.nan)

    # Adding the new rows
    results_5.append(result5_row)
    results_10.append(result10_row)
    results_15.append(result15_row)
    results_20.append(result20_row)
    results_25.append(result25_row)
    results_30.append(result30_row)


metric_cols = ['p', 'v', 'l', 'u']
estr_cols = ['ple', 'plg', 'plw', 'pls', 'pld']
columns = []
for ec in estr_cols:
    columns = columns + [ec + "_" + c for c in metric_cols]

for end_time, results_t, truth in zip([5, 10, 15, 20, 25, 30],
                                      [results_5 ,results_10, results_15, results_20, results_25, results_30],
                                      [true_5, true_10, true_15, true_20, true_25, true_30]):
    results = pd.DataFrame(results_t, columns=columns)
    for estimator in estr_cols:
        results[estimator+'_b'] = results[estimator+'_p'] - truth
        results[estimator+'_s'] = results[estimator+'_v'] ** 0.5
        results[estimator+'_c'] = np.where((results[estimator+"_l"] <= truth) & (truth <= results[estimator+'_u']),
                                           1, 0)

    # Saving simulations output
    results.to_csv("results/sim_t"+str(end_time)+"_n"+str(n_obs)+".csv")


# N=250
#          Bias    ESE    SER Coverage
# ple    -0.070  0.023  0.979    0.139
# plg    -0.016  0.032  0.986    0.887
# plw     0.002  0.034  0.977    0.937
# pls     0.006  0.036  0.966    0.935
# pld     0.001  0.038  0.968    0.939
#          Bias    ESE    SER Coverage
# ple    -0.070  0.040  0.983    0.552
# plg     0.002  0.047  0.982    0.943
# plw     0.005  0.046  0.971    0.939
# pls    -0.002  0.050  0.983    0.938
# pld     0.000  0.051  0.987    0.946
#          Bias    ESE    SER Coverage
# ple    -0.047  0.053  0.983    0.835
# plg     0.016  0.057  0.978    0.934
# plw     0.005  0.055  0.972    0.937
# pls    -0.000  0.060  0.996    0.942
# pld    -0.001  0.062  0.992    0.945
#          Bias    ESE    SER Coverage
# ple    -0.013  0.064  0.982    0.935
# plg     0.021  0.064  0.977    0.930
# plw     0.003  0.063  0.976    0.940
# pls    -0.001  0.070  0.988    0.944
# pld    -0.001  0.071  0.986    0.946
#          Bias    ESE    SER Coverage
# ple     0.027  0.072  0.981    0.926
# plg     0.013  0.073  0.979    0.938
# plw    -0.001  0.073  0.981    0.942
# pls    -0.001  0.077  0.984    0.939
# pld    -0.001  0.079  0.984    0.942
#          Bias    ESE    SER Coverage
# ple     0.069  0.079  0.979    0.843
# plg    -0.007  0.085  0.984    0.942
# plw    -0.005  0.084  0.985    0.941
# pls    -0.001  0.085  0.982    0.938
# pld    -0.001  0.085  0.981    0.941


# N = 500
#              Bias    ESE    SER   RMSE Coverage MISS-Bias
# Estimator
# ple        -0.071  0.027  1.000  0.076    0.257     0.000
# plg         0.002  0.033  0.996  0.033    0.947     0.000
# plw         0.005  0.031  0.995  0.032    0.947     0.000
# pls        -0.002  0.035  0.993  0.035    0.942     0.000
# pld         0.000  0.036  0.995  0.036    0.948     0.000
#
#              Bias    ESE    SER   RMSE Coverage MISS-Bias
# Estimator
# ple        -0.013  0.045  0.995  0.047    0.933     0.000
# plg         0.022  0.045  0.995  0.050    0.921     0.000
# plw         0.003  0.044  0.995  0.044    0.946     0.000
# pls        -0.000  0.049  0.999  0.049    0.946     0.000
# pld         0.000  0.049  1.000  0.049    0.947     0.000
#
#              Bias    ESE    SER   RMSE Coverage MISS-Bias
# Estimator
# ple         0.070  0.056  0.993  0.089    0.750     0.000
# plg        -0.006  0.060  0.997  0.060    0.946     0.000
# plw        -0.004  0.059  0.996  0.059    0.946     0.000
# pls        -0.001  0.060  0.997  0.060    0.949     0.000
# pld        -0.001  0.060  0.997  0.060    0.949     0.000

# N = 1000
#              Bias    ESE    SER   RMSE Coverage MISS-Bias
# Estimator
# ple        -0.072  0.019  0.994  0.075    0.039     0.000
# plg         0.002  0.023  1.003  0.023    0.951     0.000
# plw         0.005  0.022  1.002  0.022    0.944     0.000
# pls        -0.002  0.024  1.014  0.024    0.951     0.000
# pld         0.000  0.025  1.012  0.025    0.954     0.000
#
#              Bias    ESE    SER   RMSE Coverage MISS-Bias
# Estimator
# ple        -0.014  0.032  0.991  0.035    0.917     0.000
# plg         0.022  0.031  0.997  0.038    0.894     0.000
# plw         0.003  0.031  0.995  0.031    0.943     0.000
# pls        -0.001  0.035  0.982  0.035    0.945     0.000
# pld        -0.000  0.036  0.981  0.036    0.943     0.000
#
#              Bias    ESE    SER   RMSE Coverage MISS-Bias
# Estimator
# ple         0.069  0.040  0.989  0.080    0.577     0.000
# plg        -0.006  0.043  0.989  0.043    0.942     0.000
# plw        -0.005  0.042  0.988  0.042    0.943     0.000
# pls        -0.001  0.043  0.989  0.043    0.945     0.000
# pld        -0.001  0.043  0.989  0.043    0.945     0.000

# N = 2000
#              Bias    ESE    SER   RMSE Coverage MISS-Bias
# Estimator
# ple        -0.073  0.013  0.992  0.074    0.000     0.000
# plg         0.002  0.016  0.991  0.017    0.946     0.000
# plw         0.004  0.016  0.990  0.016    0.941     0.000
# pls        -0.002  0.017  0.990  0.018    0.943     0.000
# pld        -0.000  0.018  0.995  0.018    0.946     0.000
#
#              Bias    ESE    SER   RMSE Coverage MISS-Bias
# Estimator
# ple        -0.015  0.022  0.990  0.027    0.888     0.000
# plg         0.022  0.022  0.986  0.031    0.823     0.000
# plw         0.002  0.022  0.986  0.022    0.948     0.000
# pls        -0.001  0.025  0.983  0.025    0.946     0.000
# pld        -0.000  0.025  0.984  0.025    0.948     0.000
#
#              Bias    ESE    SER   RMSE Coverage MISS-Bias
# Estimator
# ple         0.069  0.028  0.988  0.075    0.304     0.000
# plg        -0.006  0.030  0.988  0.031    0.943     0.000
# plw        -0.004  0.030  0.989  0.030    0.946     0.000
# pls        -0.001  0.030  0.988  0.030    0.949     0.000
# pld        -0.001  0.030  0.988  0.030    0.949     0.000
