#####################################################################################
# Histograms of the 1,000 ATE estimates: 6 estimators x 3 specifications.
# Requested by Shaun (2026-07-29) to assess normality and outliers.
#
# Each panel: histogram (density), fitted normal curve, dashed line at the truth.
# Common x-axis across all panels so spread and outliers are directly comparable.
# Sources restricted to the same sim_id 1-1000 used everywhere.
#####################################################################################

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TRUTH = -0.1081508
N = 1000

BAR = '#5B8DB8'      # single muted blue for all histograms (identity is in the labels)
NORM = '#333333'     # fitted normal curve, near-black ink
TRUTHC = '#C0392B'   # truth reference line, dark red

ml = pd.read_csv('reduced_results_all.csv').sort_values('sim_id')
sc = pd.read_csv('sc_parametric_results.csv').sort_values('sim_id')


def repro(path):
    d = pd.read_csv(path).iloc[:N]
    return (d['bias'] + TRUTH).to_numpy()


DATA = {}  # (estimator, spec) -> estimates
for est, f in [('G-computation', 'gform'), ('IPW', 'iptw'), ('AIPW', 'aipw'), ('TMLE', 'tmle')]:
    DATA[(est, 'True')] = repro(f'{f}_results1.csv')
    DATA[(est, 'Main-effects')] = repro(f'{f}_results2.csv')
DATA[('SC-AIPW', 'True')] = sc['true_scaipw_rd'].to_numpy()
DATA[('SC-AIPW', 'Main-effects')] = sc['main_scaipw_rd'].to_numpy()
DATA[('SC-TMLE', 'True')] = sc['true_sctmle_rd'].to_numpy()
DATA[('SC-TMLE', 'Main-effects')] = sc['main_sctmle_rd'].to_numpy()
for est, c in [('G-computation', 'gcomp_rd'), ('IPW', 'ipw_rd'), ('AIPW', 'aipw_rd'),
               ('TMLE', 'tmle_rd'), ('SC-AIPW', 'scaipw_rd'), ('SC-TMLE', 'sctmle_rd')]:
    DATA[(est, 'Machine learning')] = ml[c].to_numpy()

ESTIMATORS = ['G-computation', 'IPW', 'AIPW', 'TMLE', 'SC-AIPW', 'SC-TMLE']
SPECS = ['True', 'Main-effects', 'Machine learning']

allv = np.concatenate(list(DATA.values()))
lo, hi = allv.min(), allv.max()
pad = 0.04 * (hi - lo)
XLIM = (lo - pad, hi + pad)
BINS = np.linspace(*XLIM, 45)

fig, axes = plt.subplots(len(ESTIMATORS), len(SPECS), figsize=(8.6, 11.4),
                         sharex=True, sharey=False)
for i, est in enumerate(ESTIMATORS):
    for j, spec in enumerate(SPECS):
        ax = axes[i, j]
        v = DATA[(est, spec)]
        ax.hist(v, bins=BINS, density=True, color=BAR, edgecolor='white', linewidth=0.3)
        m, s = np.mean(v), np.std(v, ddof=1)
        xs = np.linspace(*XLIM, 300)
        ax.plot(xs, np.exp(-0.5 * ((xs - m) / s) ** 2) / (s * np.sqrt(2 * np.pi)),
                color=NORM, linewidth=1.1)
        ax.axvline(TRUTH, color=TRUTHC, linestyle='--', linewidth=1.1)
        ax.set_yticks([])
        for side in ['left', 'top', 'right']:
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_title(spec, fontsize=10)
        if j == 0:
            ax.set_ylabel(est, fontsize=9, rotation=90)

fig.suptitle('Histograms of the 1,000 estimates of the ATE $= E(Y^1) - E(Y^0)$,'
             ' by estimator and specification\n'
             'Curve: fitted normal.  Dashed line: truth ($-0.108$).'
             '  [$E(Y^1)$ and $E(Y^0)$ histograms to follow from the g-computation studies]',
             fontsize=10)
fig.supxlabel('Estimated ATE $= E(Y^1) - E(Y^0)$ (risk difference)', fontsize=9)
fig.tight_layout(rect=[0, 0.01, 1, 0.955])
fig.savefig('histograms_ate.pdf')
fig.savefig('/tmp/histograms_ate_preview.png', dpi=70)
print('wrote histograms_ate.pdf')
