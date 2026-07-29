#####################################################################################
# Single-crossfit AIPW & TMLE with PARAMETRIC nuisance models — the two Table-3 rows
# the reduced run didn't cover ("True" and "Main-effects" specifications).
#
# - Estimators: zepid SingleCrossfitAIPTW / SingleCrossfitTMLE (n_splits=5, n_partitions=1)
# - Specs: True = Z&B's correctly specified formulas; Main = main-terms formulas
#   (identical formula strings to the authors' sim_dcaipw.py / sim_dctmle.py setups 1-2)
# - Same 1,000 datasets (sim_id 1-1000), seeded per dataset; resume-safe.
# Authors' files untouched.
#####################################################################################

import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from zepid.causal.doublyrobust import SingleCrossfitAIPTW, SingleCrossfitTMLE

BASE_SEED = 20260721   # same convention as reduced_design.py
OUT = 'sc_parametric_results.csv'

# --- formulas exactly as in the authors' sim scripts ---
G_TRUE = 'diabetes + ldl_log + ldl_160 + age_30 + age_30_sq + C(risk_score_cat)'
Q_TRUE = 'statin + statin:ldl_130 + age_sqrt + diabetes + risk_exp + ldl_120'
G_MAIN = 'diabetes + age + risk_score + ldl_log'
Q_MAIN = 'statin + diabetes + age + risk_score + ldl_log'


def logit():
    return LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)


def run_pair(dfs, g_model, q_model, seed):
    """SC-AIPW + SC-TMLE for one dataset and one specification."""
    out = {}
    try:
        m = SingleCrossfitAIPTW(dfs, 'statin', 'Y')
        m.exposure_model(g_model, logit(), bound=0.01)
        m.outcome_model(q_model, logit())
        m.fit(n_splits=5, n_partitions=1, random_state=seed)
        out['aipw_rd'], out['aipw_se'] = m.risk_difference, m.risk_difference_se
    except Exception:
        out['aipw_rd'], out['aipw_se'] = np.nan, np.nan
    try:
        m = SingleCrossfitTMLE(dfs, 'statin', 'Y')
        m.exposure_model(g_model, logit(), bound=0.01)
        m.outcome_model(q_model, logit())
        m.fit(n_splits=5, n_partitions=1, random_state=seed)
        out['tmle_rd'], out['tmle_se'] = m.risk_difference, m.risk_difference_se
    except Exception:
        out['tmle_rd'], out['tmle_se'] = np.nan, np.nan
    return out


def main():
    df = pd.read_csv('statin_sim_data.csv')
    df = df[df['sim_id'] <= 1000].copy()

    # derived columns for the correctly-specified ("True") formulas, as in the sim scripts
    df['ldl_160'] = np.where(df['ldl_log'] > np.log(160), 1, 0)
    df['age_30'] = df['age'] - 30
    df['age_30_sq'] = (df['age'] - 30) ** 2
    df['ldl_130'] = np.where(df['ldl_log'] < np.log(130), 5 - df['ldl_log'], 0)
    df['age_sqrt'] = np.sqrt(df['age'] - 39)
    df['risk_exp'] = np.exp(df['risk_score'] + 1)
    df['ldl_120'] = np.where(df['ldl_log'] > np.log(120), df['ldl_log'] ** 2, 0)

    rows, done_ids = [], set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        rows = prev.to_dict('records')
        done_ids = set(int(s) for s in prev['sim_id'])
        print(f"resuming: {len(done_ids)} datasets already done", flush=True)

    t0 = time.time()
    for sid in range(1, 1001):
        if sid in done_ids:
            continue
        dfs = df.loc[df['sim_id'] == sid].copy()
        seed = BASE_SEED + sid
        r = {'sim_id': sid}
        true_res = run_pair(dfs, G_TRUE, Q_TRUE, seed)
        r.update({f"true_sc{k}": v for k, v in true_res.items()})
        main_res = run_pair(dfs, G_MAIN, Q_MAIN, seed)
        r.update({f"main_sc{k}": v for k, v in main_res.items()})
        rows.append(r)
        done = len(rows) - len(done_ids)
        if sid % 25 == 0 or done <= 3:
            el = time.time() - t0
            print(f"sim_id {sid}: true_aipw={r['true_scaipw_rd']:+.4f} main_aipw={r['main_scaipw_rd']:+.4f}"
                  f"   [{done} this run, {el/done:.1f}s/dataset]", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)

    print(f"\nDone: {len(rows)} datasets -> {OUT} in {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
