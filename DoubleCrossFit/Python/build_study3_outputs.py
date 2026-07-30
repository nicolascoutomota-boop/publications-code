#####################################################################################
# Study 3 outputs: Table-3-format tables + histograms for E(Y^1), E(Y^0), ATE.
# Two-model g-computation, empirical-SE confidence intervals (no analytic SE exists).
#
# Truths: ATE = -0.1081508 (Z&B's value, kept for consistency with all prior tables).
# E[Y^1] = 0.2338668, E[Y^0] = 0.3419429 - computed as the mean of the DGM's outcome
# probabilities over all 6M rows (deterministic given the saved confounders; their
# difference -0.1080761 differs from Z&B's realized-draw truth by <1e-4).
#####################################################################################

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Z = 1.96
TRUTHS = {'ate': -0.1081508, 'ey1': 0.2338668, 'ey0': 0.3419429}
LABELS = {'ate': 'ATE $= E(Y^1)-E(Y^0)$', 'ey1': '$E(Y^1)$', 'ey0': '$E(Y^0)$'}
SPECS = [('true', 'True'), ('main', 'Main-effects'), ('ml', 'Machine learning')]

a = pd.read_csv('study3_results_0001_0500.csv')
b = pd.read_csv('study3_results_0501_1000.csv')
df = pd.concat([a, b], ignore_index=True).sort_values('sim_id').reset_index(drop=True)
assert len(df) == 1000 and df.sim_id.nunique() == 1000
cols = [c for c in df.columns if c != 'sim_id']
assert df[cols].isna().sum().sum() == 0, "NaNs present"

# ---------------- tables ----------------
rows = []
for est_key, est_lab in [('ey1', 'E(Y^1)'), ('ey0', 'E(Y^0)'), ('ate', 'ATE')]:
    for spec_key, spec_lab in SPECS:
        v = df[f'{spec_key}_{est_key}'].to_numpy()
        tr = TRUTHS[est_key]
        bias = v.mean() - tr
        ese = v.std(ddof=1)
        rmse = np.sqrt(bias**2 + ese**2)
        cover = np.mean((v - Z*ese < tr) & (tr < v + Z*ese))
        rows.append(dict(estimand=est_lab, spec=spec_lab, bias=bias, rmse=rmse,
                         ese=ese, cld=2*Z*ese, coverage=cover))
t = pd.DataFrame(rows)
t.to_csv('study3_tables.csv', index=False)

print("STUDY 3 (two-model g-computation), 1,000 datasets. Empirical-SE CIs; no ASE exists.")
for est_lab in ['E(Y^1)', 'E(Y^0)', 'ATE']:
    print(f"\n  {est_lab}   (truth {TRUTHS[ {'E(Y^1)':'ey1','E(Y^0)':'ey0','ATE':'ate'}[est_lab] ]:+.4f})")
    print(f"    {'spec':<18}{'Bias':>8}{'RMSE':>8}{'ESE':>8}{'CLD':>8}{'Coverage':>10}")
    for _, r in t[t.estimand == est_lab].iterrows():
        print(f"    {r.spec:<18}{r.bias:+8.3f}{r.rmse:8.3f}{r.ese:8.3f}{r.cld:8.3f}{r.coverage:9.1%}")

# ---------------- histograms: rows = estimands, cols = specs ----------------
fig, axes = plt.subplots(3, 3, figsize=(9.5, 8.5))
BAR, NORM, TRUTHC = '#5B8DB8', '#333333', '#C0392B'
for i, est_key in enumerate(['ey1', 'ey0', 'ate']):
    vals = [df[f'{s}_{est_key}'].to_numpy() for s, _ in SPECS]
    allv = np.concatenate(vals)
    pad = 0.05 * (allv.max() - allv.min())
    xlim = (min(allv.min(), TRUTHS[est_key]) - pad, max(allv.max(), TRUTHS[est_key]) + pad)
    bins = np.linspace(*xlim, 40)
    for j, ((sk, sl), v) in enumerate(zip(SPECS, vals)):
        ax = axes[i, j]
        ax.hist(v, bins=bins, density=True, color=BAR, edgecolor='white', linewidth=0.3)
        m, s = v.mean(), v.std(ddof=1)
        xs = np.linspace(*xlim, 300)
        ax.plot(xs, np.exp(-0.5*((xs-m)/s)**2)/(s*np.sqrt(2*np.pi)), color=NORM, lw=1.1)
        ax.axvline(TRUTHS[est_key], color=TRUTHC, ls='--', lw=1.1)
        ax.set_xlim(*xlim)
        ax.set_yticks([])
        for side in ['left', 'top', 'right']:
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_title(sl, fontsize=10)
        if j == 0:
            ax.set_ylabel(LABELS[est_key], fontsize=10)
fig.suptitle('Study 3 (two separate outcome models): histograms of the 1,000 g-computation estimates\n'
             'Rows: estimands.  Curve: fitted normal.  Dashed line: truth of that estimand.', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig('study3_histograms.pdf')
fig.savefig('/tmp/study3_hist_preview.png', dpi=75)
print('\nwrote study3_tables.csv and study3_histograms.pdf')
