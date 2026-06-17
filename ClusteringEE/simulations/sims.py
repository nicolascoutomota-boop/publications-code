#####################################################################################################################
# Simulation experiment for the clustering approaches
#   This script runs the experiment for all the scenarios and all the estimators described in the paper and appendix
#
# Paul Zivich (2026/06/17)
#####################################################################################################################

########################################################################
# Setup environment

# Setup dependencies
import numpy as np
import pandas as pd
from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_regression
from delicatessen.utilities import inverse_logit, aggregate_efuncs

# Setup simulation parameters
n_sims = 5000
np.random.seed(504341)


########################################################################
# Setup data generating mechanism

def dgm(n, correlation=0.):
    # Determine the size of each cluster, where n is the number of total clusters
    n_i = np.random.randint(2, 10, size=n)

    # Generating individual-level observations in each cluster (for-loop is slow, but reliable)
    d_groups = []
    for j in range(n):
        # Creating the correlation matrix for a cluster
        correlation_matrix = np.identity(n=n_i[j])
        correlation_matrix += correlation
        np.fill_diagonal(correlation_matrix, val=1)
        # Drawing the covariates from a mulitvariate-normal distribution if clustered
        d_j = pd.DataFrame()
        d_j['W'] = np.random.multivariate_normal([0, ]*n_i[j], cov=correlation_matrix, size=1)[0]
        d_j['R'] = np.random.binomial(n=1, p=inverse_logit(-d_j['W']), size=n_i[j])
        d_j['Y'] = np.random.multivariate_normal(-1 + 2*d_j['W'],
                                                 cov=correlation_matrix, size=1)[0]
        # Setting Y as missing accordingly
        d_j['Y'] = np.where(d_j['R'] == 1, d_j['Y'], np.nan)
        d_j['G'] = j + 1
        d_j['C'] = 1
        d_groups.append(d_j)
    # Return the clustered data set
    return pd.concat(d_groups, ignore_index=True)


########################################################################
# Setup estimating functions

def psi_ipw(theta):
    mu = theta[0]
    alpha = theta[1:]

    # Missingness score model
    ee_psm = ee_regression(theta=alpha, y=r, X=W, model='logistic')
    pi = inverse_logit(np.dot(W, alpha))
    ipmw = r / pi

    # IPW estimator
    ee_mu = r * ipmw * (y_no_nan - mu)
    return np.vstack([ee_mu, ee_psm])


def psi_ipw_cluster(theta):
    # Shelving the estimating functions using delicatessen utility
    psi_i = psi_ipw(theta=theta)
    return aggregate_efuncs(psi_i, group=g)


def psi_gcomp(theta):
    mu = theta[0]
    beta = theta[1:]

    # Outcome nuisance model
    ee_out = ee_regression(theta=beta, y=y_no_nan, X=W, model='linear') * r
    yhat = np.dot(W, beta)

    # G-computation estimator
    ee_mu = yhat - mu
    return np.vstack([ee_mu, ee_out])


def psi_gcomp_cluster(theta):
    # Shelving the estimating functions using delicatessen utility
    psi_i = psi_gcomp(theta=theta)
    return aggregate_efuncs(psi_i, group=g)


def psi_aipw(theta):
    mu = theta[0]
    alpha = theta[1: 1+W.shape[1]]
    beta = theta[1+W.shape[1]:]

    # Missingness score model
    ee_psm = ee_regression(theta=alpha, y=r, X=W, model='logistic')
    pi = inverse_logit(np.dot(W, alpha))
    ipmw = r / pi

    # Outcome nuisance model
    ee_out = ee_regression(theta=beta, y=y_no_nan, X=W, model='linear', weights=ipmw) * r
    yhat = np.dot(W, beta)

    # wrAIPW estimator
    ee_mu = yhat - mu
    return np.vstack([ee_mu, ee_psm, ee_out])


def psi_aipw_cluster(theta):
    # Shelving the estimating functions using delicatessen utility
    psi_i = psi_aipw(theta=theta)
    return aggregate_efuncs(psi_i, group=g)


########################################################################
# Running the full set of simulation experiments

for corr in [0., 0.25, 0.5, 0.75]:     # Specified correlation structure
    for n_cluster in [20, 100, 500]:   # Sample sizes for the clusters
        rows = []
        for i in range(n_sims):        # Running through number of iterations for the scenario
            row = []

            # Creating a specific data set
            d = dgm(n=n_cluster, correlation=corr)
            r = np.asarray(d['R'])
            y = np.asarray(d['Y'])
            y_no_nan = np.asarray(d['Y'].fillna(-999))
            W = np.asarray(d[['C', 'W']])
            g = np.asarray(d['G'])

            # Applying the estimators under consideration
            estr1 = MEstimator(psi_ipw, init=[0, 0, 0])
            estr1.estimate()
            ci = estr1.confidence_intervals()
            for x in [estr1.theta[0] + 1, estr1.variance[0, 0], ci[0, 0], ci[0, 1]]:
                row.append(x)

            estr2 = MEstimator(psi_ipw_cluster, init=[0, 0, 0])
            estr2.estimate()
            ci = estr2.confidence_intervals()
            for x in [estr2.theta[0] + 1, estr2.variance[0, 0], ci[0, 0], ci[0, 1]]:
                row.append(x)

            estr3 = MEstimator(psi_gcomp, init=[0, 0, 0])
            estr3.estimate()
            ci = estr3.confidence_intervals()
            for x in [estr3.theta[0] + 1, estr3.variance[0, 0], ci[0, 0], ci[0, 1]]:
                row.append(x)

            estr4 = MEstimator(psi_gcomp_cluster, init=[0, 0, 0])
            estr4.estimate()
            ci = estr4.confidence_intervals()
            for x in [estr4.theta[0] + 1, estr4.variance[0, 0], ci[0, 0], ci[0, 1]]:
                row.append(x)

            estr5 = MEstimator(psi_aipw, init=[0, 0, 0, 0, 0])
            estr5.estimate()
            ci = estr5.confidence_intervals()
            for x in [estr5.theta[0] + 1, estr5.variance[0, 0], ci[0, 0], ci[0, 1]]:
                row.append(x)

            estr6 = MEstimator(psi_aipw_cluster, init=[0, 0, 0, 0, 0])
            estr6.estimate()
            ci = estr6.confidence_intervals()
            for x in [estr6.theta[0] + 1, estr6.variance[0, 0], ci[0, 0], ci[0, 1]]:
                row.append(x)

            # Storing the results
            rows.append(row)

        # Processing simulation scenario results
        results = pd.DataFrame(rows, columns=['ipw_bias', 'ipw_var', 'ipw_lcl', 'ipw_ucl',
                                              'ipwc_bias', 'ipwc_var', 'ipwc_lcl', 'ipwc_ucl',
                                              'gcomp_bias', 'gcomp_var', 'gcomp_lcl', 'gcomp_ucl',
                                              'gcompc_bias', 'gcompc_var', 'gcompc_lcl', 'gcompc_ucl',
                                              'aipw_bias', 'aipw_var', 'aipw_lcl', 'aipw_ucl',
                                              'aipwc_bias', 'aipwc_var', 'aipwc_lcl', 'aipwc_ucl',
                                              ])
        for estimator in ['ipw', 'ipwc', 'gcomp', 'gcompc', 'aipw', 'aipwc']:
            results[estimator + "_cov"] = np.where((results[estimator+'_lcl'] < -1)
                                               & (-1 < results[estimator+'_ucl']), 1, 0)
            results[estimator + "_se"] = results[estimator+'_var']**0.5

        # Saving simulation scenario results as CSV
        results.to_csv("r_c"+str(int(corr*100))+"_n"+str(n_cluster)+".csv", index=False)


# END
