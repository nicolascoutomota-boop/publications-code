#####################################################################################
# Tables + histograms for studies 1 (RF-only) and 2 (n=1500): single-model
# g-computation, estimands E(Y^1), E(Y^0), ATE. Empirical-SE CIs.
# Same format as study 3's outputs (build_study3_outputs.py).
#####################################################################################

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Z = 1.96
TRUTHS = {'ate': -0.1081508, 'ey1': 0.2338668, 'ey0': 0.3419429}
LABELS = {'ate': 'ATE $= E(Y^1)-E(Y^0)$', 'ey1': '$E(Y^1)$', 'ey0': '$E(Y^0)$'}
SPECS = [('true', 'True'), ('main', 'Main-effects'), ('ml', 'Machine learning')]

STUDIES = {
    '1': dict(files=['study1_results_0001_0500.csv', 'study1_results_0501_1000.csv'],
              title='Study 1 (random forest replaces the super learner, n=3000)',
              ml_label='Random forest', out='study1'),
    '2': dict(files=['study2_results_0001_0500.csv', 'study2_results_0501_1000.csv'],
              title='Study 2 (super learner, n=1500)',
              ml_label='Machine learning', out='study2'),
}

for key, cfg in STUDIES.items():
    df = pd.concat([pd.read_csv(f) for f in cfg['files']], ignore_index=True)
    df = df.sort_values('sim_id').reset_index(drop=True)
    cols = [c for c in df.columns if c != 'sim_id']
    assert len(df) == 1000 and df.sim_id.nunique() == 1000, f"study {key}: bad row count"
    nan = df[cols].isna().sum().sum()
    assert nan == 0, f"study {key}: {nan} NaNs"

    rows = []
    print(f"\n===== {cfg['title']} — g-computation, 1,000 datasets =====")
    for est_key, est_lab in [('ey1', 'E(Y^1)'), ('ey0', 'E(Y^0)'), ('ate', 'ATE')]:
        print(f"\n  {est_lab}   (truth {TRUTHS[est_key]:+.4f})")
        print(f"    {'spec':<18}{'Bias':>8}{'RMSE':>8}{'ESE':>8}{'CLD':>8}{'Coverage':>10}")
        for spec_key, spec_lab in SPECS:
            lab = cfg['ml_label'] if spec_key == 'ml' else spec_lab
            v = df[f'{spec_key}_{est_key}'].to_numpy()
            tr = TRUTHS[est_key]
            bias, ese = v.mean() - tr, v.std(ddof=1)
            rmse = np.sqrt(bias**2 + ese**2)
            cover = np.mean((v - Z*ese < tr) & (tr < v + Z*ese))
            rows.append(dict(estimand=est_lab, spec=lab, bias=bias, rmse=rmse,
                             ese=ese, cld=2*Z*ese, coverage=cover))
            print(f"    {lab:<18}{bias:+8.3f}{rmse:8.3f}{ese:8.3f}{2*Z*ese:8.3f}{cover:9.1%}")
    pd.DataFrame(rows).to_csv(f"{cfg['out']}_tables.csv", index=False)

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
                ax.set_title(cfg['ml_label'] if sk == 'ml' else sl, fontsize=10)
            if j == 0:
                ax.set_ylabel(LABELS[est_key], fontsize=10)
    fig.suptitle(f"{cfg['title']}:\nhistograms of the 1,000 g-computation estimates. "
                 "Curve: fitted normal. Dashed line: truth.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(f"{cfg['out']}_histograms.pdf")
    fig.savefig(f"/tmp/{cfg['out']}_hist_preview.png", dpi=75)
    print(f"\n  wrote {cfg['out']}_tables.csv and {cfg['out']}_histograms.pdf")
