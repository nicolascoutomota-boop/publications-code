#####################################################################################
# Reduced-design driver (Shaun's proposal, agreed 2026-07-21)
#
# Runs SIX methods on each simulated dataset and stores every ATE point estimate:
#   1. IPW                       (authors' IPTW,   super learner)
#   2. g-computation NO bootstrap (authors' GFormula, super learner)  -> point est only
#   3. AIPW, no cross-fitting     (authors' AIPTW,  super learner)
#   4. TMLE, no cross-fitting     (authors' TMLE,   super learner)
#   5. AIPW, 5-fold single cross-fitting, ONE split  (zepid SingleCrossfitAIPTW)
#   6. TMLE, 5-fold single cross-fitting, ONE split  (zepid SingleCrossfitTMLE)
#
# Design decisions (see PROJECT_LOG.md): 1,000 datasets; single split (n_partitions=1);
# super learner left exactly as published (K=10, neural net + 500-tree RF kept).
# A seed is set PER DATASET so the whole run is reproducible and chunkable across cores.
#
# This file does NOT modify any of the authors' source files. It imports their estimator
# classes and zepid's single-cross-fit classes and drives them.
#####################################################################################

import argparse
import os
import time
import numpy as np
import pandas as pd

from estimators import IPTW, GFormula, AIPTW, TMLE
from super_learner import superlearnersetup
from zepid.causal.doublyrobust import SingleCrossfitAIPTW, SingleCrossfitTMLE

TRUTH = -0.1081508
BASE_SEED = 20260721  # per-dataset seed = BASE_SEED + sim_id  (reproducible + chunkable)

# Main-terms formulas (identical to the authors' setup==3 machine-learning specification)
G_MODEL = 'diabetes + age + risk_score + ldl_log'
Q_MODEL = 'statin + diabetes + age + risk_score + ldl_log'


RF_JOBS = 1  # set from --rf-jobs; PURE SPEED FLAG (parallel tree building), no effect on the
             # statistical method. Applied at runtime to the estimator object; the authors'
             # super_learner.py is never edited.


def sl(K):
    """A fresh super-learner instance (binary), as published, with only n_jobs adjusted."""
    s = superlearnersetup(var_type='binary', K=K)
    if RF_JOBS != 1:
        from sklearn.ensemble import RandomForestClassifier
        for est in s.library:
            if isinstance(est, RandomForestClassifier):
                est.set_params(n_jobs=RF_JOBS)
    return s


def run_one_dataset(dfs, K, seed):
    """Return a dict of the six point estimates (+ SEs where available) for one dataset."""
    row = {}

    # ---- 1. IPW ------------------------------------------------------------
    np.random.seed(seed)
    try:
        m = IPTW(dfs, 'statin', 'Y')
        m.treatment_model(G_MODEL, sl(K), bound=0.01)
        m.fit()
        row['ipw_rd'], row['ipw_se'] = m.risk_difference, m.risk_difference_se
    except Exception:
        row['ipw_rd'], row['ipw_se'] = np.nan, np.nan

    # ---- 2. g-computation, NO bootstrap (point estimate only) --------------
    np.random.seed(seed)
    try:
        m = GFormula(dfs, treatment='statin', outcome='Y')
        m.outcome_model(covariates=Q_MODEL, estimator=sl(K))
        m.fit()
        row['gcomp_rd'] = m.risk_difference        # no SE by design; empirical SE used later
    except Exception:
        row['gcomp_rd'] = np.nan

    # ---- 3. AIPW, no cross-fitting ----------------------------------------
    np.random.seed(seed)
    try:
        m = AIPTW(dfs, 'statin', 'Y')
        m.treatment_model(G_MODEL, sl(K), bound=0.01)
        m.outcome_model(Q_MODEL, sl(K))
        m.fit()
        row['aipw_rd'], row['aipw_se'] = m.risk_difference, m.risk_difference_se
    except Exception:
        row['aipw_rd'], row['aipw_se'] = np.nan, np.nan

    # ---- 4. TMLE, no cross-fitting ----------------------------------------
    np.random.seed(seed)
    try:
        m = TMLE(dfs, 'statin', 'Y')
        m.treatment_model(G_MODEL, sl(K), bound=0.01)
        m.outcome_model(Q_MODEL, sl(K))
        m.fit()
        row['tmle_rd'], row['tmle_se'] = m.risk_difference, m.risk_difference_se
    except Exception:
        row['tmle_rd'], row['tmle_se'] = np.nan, np.nan

    # ---- 5. AIPW, 5-fold single cross-fit, ONE split ----------------------
    try:
        m = SingleCrossfitAIPTW(dfs, 'statin', 'Y')
        m.exposure_model(G_MODEL, sl(K), bound=0.01)
        m.outcome_model(Q_MODEL, sl(K))
        m.fit(n_splits=5, n_partitions=1, random_state=seed)
        row['scaipw_rd'], row['scaipw_se'] = m.risk_difference, m.risk_difference_se
    except Exception:
        row['scaipw_rd'], row['scaipw_se'] = np.nan, np.nan

    # ---- 6. TMLE, 5-fold single cross-fit, ONE split ----------------------
    try:
        m = SingleCrossfitTMLE(dfs, 'statin', 'Y')
        m.exposure_model(G_MODEL, sl(K), bound=0.01)
        m.outcome_model(Q_MODEL, sl(K))
        m.fit(n_splits=5, n_partitions=1, random_state=seed)
        row['sctmle_rd'], row['sctmle_se'] = m.risk_difference, m.risk_difference_se
    except Exception:
        row['sctmle_rd'], row['sctmle_se'] = np.nan, np.nan

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=1, help='first sim_id (inclusive)')
    ap.add_argument('--end', type=int, default=1000, help='last sim_id (inclusive)')
    ap.add_argument('--k', type=int, default=10, help='super-learner CV folds (published = 10)')
    ap.add_argument('--out', type=str, default=None, help='output CSV path')
    ap.add_argument('--rf-jobs', type=int, default=1, help='RandomForest n_jobs (speed only)')
    args = ap.parse_args()

    global RF_JOBS
    RF_JOBS = args.rf_jobs

    out = args.out or f'reduced_results_{args.start}_{args.end}.csv'

    # Resume: if the output already has some datasets, skip them (survives interruptions)
    rows = []
    done_ids = set()
    if os.path.exists(out):
        prev = pd.read_csv(out)
        rows = prev.to_dict('records')
        done_ids = set(int(s) for s in prev['sim_id'])
        print(f"resuming {out}: {len(done_ids)} datasets already done", flush=True)

    df = pd.read_csv('statin_sim_data.csv')
    t0 = time.time()
    for sid in range(args.start, args.end + 1):
        if sid in done_ids:
            continue
        dfs = df.loc[df['sim_id'] == sid].copy()
        seed = BASE_SEED + sid
        r = run_one_dataset(dfs, K=args.k, seed=seed)
        r['sim_id'] = sid
        rows.append(r)
        el = time.time() - t0
        done = len(rows) - len(done_ids)   # datasets computed this run
        print(f"sim_id {sid}: "
              + " ".join(f"{k}={r[k]:+.4f}" for k in
                         ['ipw_rd', 'gcomp_rd', 'aipw_rd', 'tmle_rd', 'scaipw_rd', 'sctmle_rd'])
              + f"   [{done} done, {el/done:.1f}s/dataset]", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)   # write after every dataset (crash-safe)

    print(f"\nDone: {len(rows)} datasets -> {out} in {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
