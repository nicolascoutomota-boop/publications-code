#####################################################################################
# STUDY 3 (Shaun, 2026-07-29): g-computation with TWO separate outcome models,
# one for E(Y|X=1,Z) fit on the treated, one for E(Y|X=0,Z) fit on the untreated.
# G-computation ONLY. Records E(Y^1), E(Y^0) and the ATE per dataset.
#
# Three specifications, mirroring Z&B's Table 3 rows:
#   true : correctly specified parametric models (arm-specific versions of their formulas;
#          the treated-arm model includes ldl_130, which in Z&B's single model enters
#          only through the statin:ldl_130 interaction, i.e. only for the treated)
#   main : main-terms logistic regression, both arms
#   ml   : super learner (K=10, exactly as published), both arms
#
# The two-model structure itself is new driver code (their GFormula fits one model with
# treatment as a covariate); the authors' files are untouched.
# Seeded per dataset (BASE_SEED + sim_id); resume-safe; chunkable via --start/--end.
#####################################################################################

import argparse
import os
import time
import numpy as np
import pandas as pd
import patsy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from super_learner import superlearnersetup

BASE_SEED = 20260721

# Arm-specific outcome formulas per specification (no treatment term: each arm has its own model)
SPECS = {
    'true': {'treated': 'ldl_130 + age_sqrt + diabetes + risk_exp + ldl_120',
             'untreated': 'age_sqrt + diabetes + risk_exp + ldl_120'},
    'main': {'treated': 'diabetes + age + risk_score + ldl_log',
             'untreated': 'diabetes + age + risk_score + ldl_log'},
    'ml':   {'treated': 'diabetes + age + risk_score + ldl_log',
             'untreated': 'diabetes + age + risk_score + ldl_log'},
}

RF_JOBS = 1


def make_estimator(spec, K):
    if spec == 'ml':
        s = superlearnersetup(var_type='binary', K=K)
        if RF_JOBS != 1:
            for est in s.library:
                if isinstance(est, RandomForestClassifier):
                    est.set_params(n_jobs=RF_JOBS)
        return s
    return LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)


def arm_mean(dfs, arm_df, formula, estimator):
    """Fit the outcome model on one arm, predict everyone, return the mean prediction."""
    X_fit = np.asarray(patsy.dmatrix(formula + ' - 1', arm_df))
    y_fit = np.asarray(arm_df['Y'])
    fm = estimator.fit(X=X_fit, y=y_fit)
    X_all = np.asarray(patsy.dmatrix(formula + ' - 1', dfs))
    if hasattr(fm, 'predict_proba'):
        pred = fm.predict_proba(X_all)[:, 1]
    else:
        pred = fm.predict(X_all)
    return float(np.mean(pred))


def run_one(dfs, K, seed):
    row = {}
    treated = dfs[dfs['statin'] == 1]
    untreated = dfs[dfs['statin'] == 0]
    for spec, formulas in SPECS.items():
        np.random.seed(seed)
        try:
            ey1 = arm_mean(dfs, treated, formulas['treated'], make_estimator(spec, K))
            ey0 = arm_mean(dfs, untreated, formulas['untreated'], make_estimator(spec, K))
            row[f'{spec}_ey1'], row[f'{spec}_ey0'], row[f'{spec}_ate'] = ey1, ey0, ey1 - ey0
        except Exception:
            row[f'{spec}_ey1'] = row[f'{spec}_ey0'] = row[f'{spec}_ate'] = np.nan
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=1000)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--rf-jobs', type=int, default=1)
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    global RF_JOBS
    RF_JOBS = args.rf_jobs
    out = args.out or f'study3_results_{args.start}_{args.end}.csv'

    rows, done_ids = [], set()
    if os.path.exists(out):
        prev = pd.read_csv(out)
        rows = prev.to_dict('records')
        done_ids = set(int(s) for s in prev['sim_id'])
        print(f"resuming {out}: {len(done_ids)} datasets already done", flush=True)

    df = pd.read_csv('statin_sim_data.csv')
    df = df[(df['sim_id'] >= args.start) & (df['sim_id'] <= args.end)].copy()
    # derived columns for the 'true' formulas (same construction as the authors' scripts)
    df['ldl_130'] = np.where(df['ldl_log'] < np.log(130), 5 - df['ldl_log'], 0)
    df['age_sqrt'] = np.sqrt(df['age'] - 39)
    df['risk_exp'] = np.exp(df['risk_score'] + 1)
    df['ldl_120'] = np.where(df['ldl_log'] > np.log(120), df['ldl_log'] ** 2, 0)

    t0 = time.time()
    for sid in range(args.start, args.end + 1):
        if sid in done_ids:
            continue
        dfs = df[df['sim_id'] == sid].copy()
        r = {'sim_id': sid}
        r.update(run_one(dfs, K=args.k, seed=BASE_SEED + sid))
        rows.append(r)
        done = len(rows) - len(done_ids)
        el = time.time() - t0
        print(f"sim_id {sid}: ml_ate={r['ml_ate']:+.4f} ml_ey1={r['ml_ey1']:.4f} "
              f"ml_ey0={r['ml_ey0']:.4f}   [{done} this run, {el/done:.1f}s/dataset]", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)

    print(f"\nDone: {len(rows)} datasets -> {out} in {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
