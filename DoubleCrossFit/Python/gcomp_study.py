#####################################################################################
# Studies 1 and 2 (Shaun, g-computation only), single-model g-computation using the
# AUTHORS' OWN GFormula class, recording E(Y^1), E(Y^0) and the ATE per dataset.
#
#   Study 1: --ml rf          random forest replaces the super learner (n=3000)
#   Study 2: --ml sl --n 1500 super learner, first 1,500 rows of each dataset
#
# Three specifications per dataset (true / main-effects / ml), seeded, resume-safe.
#####################################################################################

import argparse
import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from estimators import GFormula
from super_learner import superlearnersetup

BASE_SEED = 20260721
Q_TRUE = 'statin + statin:ldl_130 + age_sqrt + diabetes + risk_exp + ldl_120'
Q_MAIN = 'statin + diabetes + age + risk_score + ldl_log'

RF_JOBS = 1
ML_KIND = 'sl'
K = 10


def ml_estimator():
    if ML_KIND == 'rf':
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=20, n_jobs=RF_JOBS)
    s = superlearnersetup(var_type='binary', K=K)
    if RF_JOBS != 1:
        for est in s.library:
            if isinstance(est, RandomForestClassifier):
                est.set_params(n_jobs=RF_JOBS)
    return s


def logit():
    return LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)


def run_one(dfs, seed):
    row = {}
    for spec, q_model, est_fn in [('true', Q_TRUE, logit), ('main', Q_MAIN, logit),
                                  ('ml', Q_MAIN, ml_estimator)]:
        np.random.seed(seed)
        try:
            g = GFormula(dfs, treatment='statin', outcome='Y')
            g.outcome_model(covariates=q_model, estimator=est_fn())
            g.fit()
            row[f'{spec}_ey1'] = g.risk_all
            row[f'{spec}_ey0'] = g.risk_none
            row[f'{spec}_ate'] = g.risk_difference
        except Exception:
            row[f'{spec}_ey1'] = row[f'{spec}_ey0'] = row[f'{spec}_ate'] = np.nan
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=1000)
    ap.add_argument('--n', type=int, default=3000, help='rows per dataset (1500 = first half)')
    ap.add_argument('--ml', choices=['sl', 'rf'], default='sl')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--rf-jobs', type=int, default=1)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    global RF_JOBS, ML_KIND, K
    RF_JOBS, ML_KIND, K = args.rf_jobs, args.ml, args.k

    rows, done_ids = [], set()
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        rows = prev.to_dict('records')
        done_ids = set(int(s) for s in prev['sim_id'])
        print(f"resuming {args.out}: {len(done_ids)} done", flush=True)

    df = pd.read_csv('statin_sim_data.csv')
    df = df[(df['sim_id'] >= args.start) & (df['sim_id'] <= args.end)].copy()
    df['ldl_130'] = np.where(df['ldl_log'] < np.log(130), 5 - df['ldl_log'], 0)
    df['age_sqrt'] = np.sqrt(df['age'] - 39)
    df['risk_exp'] = np.exp(df['risk_score'] + 1)
    df['ldl_120'] = np.where(df['ldl_log'] > np.log(120), df['ldl_log'] ** 2, 0)

    t0 = time.time()
    for sid in range(args.start, args.end + 1):
        if sid in done_ids:
            continue
        dfs = df[df['sim_id'] == sid]
        if args.n < len(dfs):
            dfs = dfs.iloc[:args.n]          # first n rows: independent draws
        dfs = dfs.copy()
        r = {'sim_id': sid}
        r.update(run_one(dfs, BASE_SEED + sid))
        rows.append(r)
        done = len(rows) - len(done_ids)
        el = time.time() - t0
        print(f"sim_id {sid}: ml_ate={r['ml_ate']:+.4f} ml_ey1={r['ml_ey1']:.4f} "
              f"ml_ey0={r['ml_ey0']:.4f}   [{done} this run, {el/done:.1f}s/dataset]", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    print(f"\nDone: {len(rows)} datasets -> {args.out} in {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
