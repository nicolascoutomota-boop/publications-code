#####################################################################################
# E(Y^0) studies (3c / 4 / 5): Table-3-format summary from e0_studies.py output.
#
# Six estimators x three specs, E(Y^0) only. Truth E[Y^0] = 0.3419429 (same value
# as build_study3_outputs.py: mean of the DGM's outcome probabilities over all 6M
# rows). Coverage is reported two ways, as in the reduced design:
#   cover_own  - CIs from each dataset's own (influence-function) SE
#   cover_emp  - CIs from the empirical SE across datasets (the only option for
#                g-computation, which has no analytic SE here)
# Rows with missing values for an estimator (pygam NaN folds, e.g. 3c sims 154/831
# for the SC methods) are dropped for that estimator only; n_used is reported.
#
# Usage: python build_e0_outputs.py --prefix study3c
#####################################################################################

import argparse
import numpy as np
import pandas as pd

Z = 1.96
TRUTH_EY0 = 0.3419429
ESTIMATORS = [('gcomp', 'G-computation'), ('ipw', 'IPW (Hajek)'),
              ('aipw', 'AIPW'), ('tmle', 'TMLE'),
              ('scaipw', 'SC-AIPW'), ('sctmle', 'SC-TMLE')]
SPECS = [('true', 'True'), ('main', 'Main-effects'), ('ml', 'Machine learning')]

ap = argparse.ArgumentParser()
ap.add_argument('--prefix', required=True, help='e.g. study3c, study4, study5')
args = ap.parse_args()

a = pd.read_csv(f'{args.prefix}_results_0001_0500.csv')
b = pd.read_csv(f'{args.prefix}_results_0501_1000.csv')
df = pd.concat([a, b], ignore_index=True).sort_values('sim_id').reset_index(drop=True)
assert df.sim_id.nunique() == len(df)
print(f"{args.prefix}: {len(df)} datasets")

rows = []
for spec_key, spec_lab in SPECS:
    for est_key, est_lab in ESTIMATORS:
        v = df[f'{spec_key}_{est_key}_e0']
        se = df[f'{spec_key}_{est_key}_se']
        ok = v.notna()
        v = v[ok].to_numpy()
        bias = v.mean() - TRUTH_EY0
        ese = v.std(ddof=1)
        rmse = np.sqrt(bias**2 + ese**2)
        cover_emp = np.mean((v - Z * ese < TRUTH_EY0) & (TRUTH_EY0 < v + Z * ese))
        if se[ok].notna().all():
            sev = se[ok].to_numpy()
            ase = sev.mean()
            cover_own = np.mean((v - Z * sev < TRUTH_EY0) & (TRUTH_EY0 < v + Z * sev))
        else:
            ase, cover_own = np.nan, np.nan
        rows.append(dict(spec=spec_lab, estimator=est_lab, n_used=int(ok.sum()),
                         bias=bias, ese=ese, ase=ase, rmse=rmse,
                         cover_own=cover_own, cover_emp=cover_emp))

t = pd.DataFrame(rows)
out = f'{args.prefix}_tables.csv'
t.round({'bias': 4, 'ese': 4, 'ase': 4, 'rmse': 4,
         'cover_own': 3, 'cover_emp': 3}).to_csv(out, index=False)

print(f"E(Y^0) only, truth {TRUTH_EY0:+.4f}. cover_own = own-SE CIs; "
      f"cover_emp = empirical-SE CIs.")
for spec_key, spec_lab in SPECS:
    print(f"\n  {spec_lab}")
    print(f"    {'estimator':<15}{'n':>5}{'Bias':>9}{'ESE':>8}{'ASE':>8}"
          f"{'cov own':>9}{'cov emp':>9}")
    for _, r in t[t.spec == spec_lab].iterrows():
        ase = f"{r.ase:8.4f}" if np.isfinite(r.ase) else '       -'
        co = f"{r.cover_own:8.1%}" if np.isfinite(r.cover_own) else '       -'
        print(f"    {r.estimator:<15}{r.n_used:>5}{r.bias:+9.4f}{r.ese:8.4f}{ase}"
              f"{co}{r.cover_emp:9.1%}")
print(f"\nwrote {out}")
