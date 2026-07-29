#####################################################################################
# Build our reproduction of Zivich & Breskin's Table 3 (with Shaun's two changes).
#
# Structure mirrors the paper's Table 3: 6 estimators x 3 specifications
# (True / Main-effects / Machine learning), columns Bias, RMSE, ASE, ESE, CLD, Coverage.
#
# Shaun's changes:
#  1. G-computation uses the EMPIRICAL SE (SD of the point estimates) for its CIs
#     in every panel (no bootstrap). Its ASE is therefore "--".
#  2. Single cross-fit AIPW/TMLE replace the double cross-fit versions.
#
# Sources (all restricted to the same sim_id 1-1000):
#  - Machine learning panel: reduced_results_all.csv          (the finished reduced run)
#  - True/Main panels, gform/iptw/aipw/tmle: *_results1.csv / *_results2.csv
#      (earlier runs of the authors' unmodified code; row i = sim_id i+1)
#  - True/Main panels, SC-AIPW/SC-TMLE: sc_parametric_results.csv
#####################################################################################

import numpy as np
import pandas as pd

TRUTH = -0.1081508
Z = 1.96
N = 1000


def summarise_est_se(est, se=None, empirical_ci=False):
    """Return the six Table-3 numbers from per-dataset estimates (and SEs)."""
    est = np.asarray(est, dtype=float)[:N]
    bias = np.nanmean(est) - TRUTH
    ese = np.nanstd(est, ddof=1)
    rmse = np.sqrt(bias**2 + ese**2)
    if empirical_ci or se is None:
        # CI = est +/- z * ESE  (Shaun's "true SE" convention)
        cover = np.nanmean((est - Z*ese < TRUTH) & (TRUTH < est + Z*ese))
        return bias, rmse, np.nan, ese, 2*Z*ese, cover
    se = np.asarray(se, dtype=float)[:N]
    ase = np.nanmean(se)
    cover = np.nanmean((est - Z*se < TRUTH) & (TRUTH < est + Z*se))
    return bias, rmse, ase, ese, 2*Z*ase, cover


def from_repro_csv(path, empirical_ci=False):
    """Earlier reproduction CSVs store bias/std per dataset; recover estimates."""
    d = pd.read_csv(path).iloc[:N]
    est = d['bias'].to_numpy() + TRUTH
    return summarise_est_se(est, d['std'].to_numpy(), empirical_ci=empirical_ci)


rows = []  # (estimator, spec, six numbers)

# ---------------- True and Main-effects panels ----------------
for spec, tag in [('True', '1'), ('Main-effects', '2')]:
    rows.append(('G-computation', spec, from_repro_csv(f'gform_results{tag}.csv', empirical_ci=True)))
    rows.append(('IPW',           spec, from_repro_csv(f'iptw_results{tag}.csv')))
    rows.append(('AIPW',          spec, from_repro_csv(f'aipw_results{tag}.csv')))
    rows.append(('TMLE',          spec, from_repro_csv(f'tmle_results{tag}.csv')))

sc = pd.read_csv('sc_parametric_results.csv').sort_values('sim_id')
for spec, pre in [('True', 'true_sc'), ('Main-effects', 'main_sc')]:
    rows.append(('SC-AIPW', spec, summarise_est_se(sc[f'{pre}aipw_rd'], sc[f'{pre}aipw_se'])))
    rows.append(('SC-TMLE', spec, summarise_est_se(sc[f'{pre}tmle_rd'], sc[f'{pre}tmle_se'])))

# ---------------- Machine-learning panel ----------------
ml = pd.read_csv('reduced_results_all.csv').sort_values('sim_id')
rows.append(('G-computation', 'Machine learning', summarise_est_se(ml['gcomp_rd'], empirical_ci=True)))
rows.append(('IPW',  'Machine learning', summarise_est_se(ml['ipw_rd'],  ml['ipw_se'])))
rows.append(('AIPW', 'Machine learning', summarise_est_se(ml['aipw_rd'], ml['aipw_se'])))
rows.append(('TMLE', 'Machine learning', summarise_est_se(ml['tmle_rd'], ml['tmle_se'])))
rows.append(('SC-AIPW', 'Machine learning', summarise_est_se(ml['scaipw_rd'], ml['scaipw_se'])))
rows.append(('SC-TMLE', 'Machine learning', summarise_est_se(ml['sctmle_rd'], ml['sctmle_se'])))

# ---------------- output, ordered like the paper's Table 3 ----------------
order = ['G-computation', 'IPW', 'AIPW', 'TMLE', 'SC-AIPW', 'SC-TMLE']
spec_order = ['True', 'Main-effects', 'Machine learning']

out = pd.DataFrame(
    [(e, s, *v) for e in order for s in spec_order
     for (er, sr, v) in [next((r for r in rows if r[0] == e and r[1] == s))] ],
    columns=['Estimator', 'Specification', 'Bias', 'RMSE', 'ASE', 'ESE', 'CLD', 'Coverage'])
out.to_csv('table3_reproduction.csv', index=False)

print(f"Our Table 3 (n=1000 datasets, sim_id 1-1000; truth {TRUTH})")
print("G-computation CIs from empirical SE in all panels; SC = single cross-fit (5 splits, 1 partition)\n")
cur = None
print(f"{'':22}{'Bias':>8}{'RMSE':>8}{'ASE':>8}{'ESE':>8}{'CLD':>8}{'Coverage':>10}")
for _, r in out.iterrows():
    if r['Estimator'] != cur:
        cur = r['Estimator']
        print(cur)
    ase = '   --' if np.isnan(r['ASE']) else f"{r['ASE']:.3f}"
    print(f"  {r['Specification']:<20}{r['Bias']:+8.3f}{r['RMSE']:8.3f}{ase:>8}{r['ESE']:8.3f}"
          f"{r['CLD']:8.3f}{r['Coverage']:9.1%}")
print("\nWrote table3_reproduction.csv")
