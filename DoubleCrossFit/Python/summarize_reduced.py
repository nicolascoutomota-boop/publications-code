#####################################################################################
# Summarise the reduced-design run (Shaun's 6-method design, 1,000 datasets).
#
# Produces, per method: mean bias, empirical SE (ESE = SD of the 1,000 point estimates,
# i.e. Shaun's "true SE"), average estimated SE (ASE), RMSE, and TWO coverages:
#   - own-SE coverage: CI = est +/- 1.96 * (method's own SE)   [not available for g-comp]
#   - true-SE coverage: CI = est +/- 1.96 * ESE                [available for all six]
# Also merges the two chunk CSVs into one file with all 1,000 x 6 point estimates.
#####################################################################################

import numpy as np
import pandas as pd

TRUTH = -0.1081508
Z = 1.96

# Load + merge the two chunks
a = pd.read_csv('reduced_results_0001_0500.csv')
b = pd.read_csv('reduced_results_0501_1000.csv')
df = pd.concat([a, b], ignore_index=True).sort_values('sim_id').reset_index(drop=True)
df.to_csv('reduced_results_all.csv', index=False)

# Integrity checks
assert len(df) == 1000, f"expected 1000 datasets, got {len(df)}"
assert df['sim_id'].nunique() == 1000, "duplicate/missing sim_ids"
assert set(df['sim_id']) == set(range(1, 1001)), "sim_ids not exactly 1..1000"

METHODS = [
    ('IPW',                    'ipw_rd',    'ipw_se'),
    ('G-computation (no BS)',  'gcomp_rd',  None),      # no per-dataset SE by design
    ('AIPW',                   'aipw_rd',   'aipw_se'),
    ('TMLE',                   'tmle_rd',   'tmle_se'),
    ('AIPW 5-fold cross-fit',  'scaipw_rd', 'scaipw_se'),
    ('TMLE 5-fold cross-fit',  'sctmle_rd', 'sctmle_se'),
]

rows = []
for name, rd_col, se_col in METHODS:
    est = df[rd_col].to_numpy()
    n_nan = int(np.isnan(est).sum())
    bias = np.nanmean(est) - TRUTH
    ese = np.nanstd(est, ddof=1)                      # empirical / "true" SE
    rmse = np.sqrt(bias**2 + ese**2)

    # true-SE coverage (all methods): CI uses the empirical SE
    lo_t, hi_t = est - Z*ese, est + Z*ese
    cov_true = np.nanmean((lo_t < TRUTH) & (TRUTH < hi_t))

    if se_col is not None:
        se = df[se_col].to_numpy()
        ase = np.nanmean(se)
        lo_o, hi_o = est - Z*se, est + Z*se
        cov_own = np.nanmean((lo_o < TRUTH) & (TRUTH < hi_o))
    else:
        ase, cov_own = np.nan, np.nan

    rows.append(dict(method=name, n_nan=n_nan, bias=bias, ese=ese, ase=ase,
                     rmse=rmse, cover_own=cov_own, cover_true=cov_true))

summary = pd.DataFrame(rows)
summary.to_csv('reduced_summary.csv', index=False)

pd.set_option('display.width', 160, 'display.max_columns', 20)
print(f"Reduced design — {len(df)} datasets, truth = {TRUTH}\n")
fmt = "{:<24} {:>6} {:>9} {:>9} {:>9} {:>9} {:>10} {:>11}"
print(fmt.format("method", "NaNs", "bias", "ESE", "ASE", "RMSE", "cov(ownSE)", "cov(trueSE)"))
print("-"*100)
for r in rows:
    print(fmt.format(
        r['method'], r['n_nan'],
        f"{r['bias']:+.4f}", f"{r['ese']:.4f}",
        "  --  " if np.isnan(r['ase']) else f"{r['ase']:.4f}",
        f"{r['rmse']:.4f}",
        "  --  " if np.isnan(r['cover_own']) else f"{r['cover_own']:.3f}",
        f"{r['cover_true']:.3f}"))
print("\nWrote reduced_results_all.csv (all point estimates) and reduced_summary.csv")
