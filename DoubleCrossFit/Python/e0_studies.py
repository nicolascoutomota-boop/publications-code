#####################################################################################
# Studies 3-continued / 4 / 5 (Shaun, 2026-07-30): estimating E(Y^0) ONLY.
#
# Design rules (his email):
#   - separate outcome models per arm; E(Y|X=1,Z) is NOT estimated at all
#   - NO individual with X=1 is ever used to fit E(Y|X=0,Z)
#   - nuisances fitted ONCE per dataset and SHARED across estimators:
#       1 propensity model (full sample) + 1 untreated-only outcome model (full sample)
#       + 5 fold propensity models + 5 fold untreated-only outcome models  = 12 ML fits
#   - estimators: g-comp, IPW (Hajek), AIPW, TMLE, SC-AIPW, SC-TMLE (5 folds, 1 partition)
#
# Studies:  3c: --ml sl --n 3000   |   4: --ml sl --n 1500   |   5: --ml rf --n 3000
# Own SEs from influence functions; empirical-SE coverage computed at table time.
#####################################################################################

import argparse
import os
import time
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy.stats import logistic
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from super_learner import superlearnersetup

BASE_SEED = 20260721
BOUND = 0.01          # propensity bounded to [0.01, 0.99], as everywhere in this project
QB = 1e-6             # bound on outcome predictions before logit (TMLE offset)

SPECS = {
    'true': {'g': 'diabetes + ldl_log + ldl_160 + age_30 + age_30_sq + C(risk_score_cat)',
             'm0': 'age_sqrt + diabetes + risk_exp + ldl_120'},
    'main': {'g': 'diabetes + age + risk_score + ldl_log',
             'm0': 'diabetes + age + risk_score + ldl_log'},
    'ml':   {'g': 'diabetes + age + risk_score + ldl_log',
             'm0': 'diabetes + age + risk_score + ldl_log'},
}

RF_JOBS, ML_KIND, K = 1, 'sl', 10


def ml_estimator():
    if ML_KIND == 'rf':
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=20, n_jobs=RF_JOBS)
    s = superlearnersetup(var_type='binary', K=K)
    if RF_JOBS != 1:
        for est in s.library:
            if isinstance(est, RandomForestClassifier):
                est.set_params(n_jobs=RF_JOBS)
    return s


def logit_est():
    return LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)


def fit_predict(formula, fit_df, y, predict_df, est_factory, retries=3, fallback=None):
    """Fit a fresh estimator on fit_df rows; predict predict_df rows.

    The authors' SuperLearner uses an UNSHUFFLED KFold, so its SLSQP weight step can
    fail persistently for a given row order. Retries therefore PERMUTE the row order
    (changing fold composition; the estimator itself is unchanged). If every attempt
    fails and a fallback factory is given (logistic regression), use it and log loudly.
    """
    Xf = np.asarray(patsy.dmatrix(formula + ' - 1', fit_df))
    Xp = np.asarray(patsy.dmatrix(formula + ' - 1', predict_df))
    yv = np.asarray(y)
    last = None
    for attempt in range(retries + 1):
        try:
            if attempt == 0:
                Xa, ya = Xf, yv
            else:
                perm = np.random.permutation(len(yv))
                Xa, ya = Xf[perm], yv[perm]
            fm = est_factory().fit(X=Xa, y=ya)
            if hasattr(fm, 'predict_proba'):
                return fm.predict_proba(Xp)[:, 1]
            return fm.predict(Xp)
        except Exception as e:
            last = e
            print(f"    fit retry {attempt+1} (row-shuffled) after: "
                  f"{type(e).__name__}: {e}", flush=True)
    if fallback is not None:
        print("    FALLBACK: logistic regression used for this single nuisance fit "
              "(super learner failed all attempts)", flush=True)
        fm = fallback().fit(X=Xf, y=yv)
        return fm.predict_proba(Xp)[:, 1]
    raise last


def bound_p(p, lo, hi):
    return np.clip(p, lo, hi)


# ---------------- E(Y^0) estimators, given nuisance predictions ----------------
def est_gcomp(A, Y, pi, m0):
    psi = m0.mean()
    return psi, np.nan

def est_ipw(A, Y, pi, m0):
    w = (1 - A) / (1 - pi)
    psi = np.sum(w * Y) / np.sum(w)
    ic = w * (Y - psi) / w.mean()
    return psi, ic.std(ddof=1) / np.sqrt(len(Y))

def est_aipw(A, Y, pi, m0):
    po = m0 + (1 - A) * (Y - m0) / (1 - pi)
    return po.mean(), po.std(ddof=1) / np.sqrt(len(Y))

def _tmle_update(A, Y, pi, m0):
    """One-parameter targeting for E(Y^0): logistic fluctuation with clever covariate."""
    m0b = bound_p(m0, QB, 1 - QB)
    H0 = (1 - A) / (1 - pi)
    off = np.log(m0b / (1 - m0b))
    fit = sm.GLM(Y, H0.reshape(-1, 1), offset=off, family=sm.families.Binomial(),
                 missing='drop').fit()
    eps = fit.params[0]
    return logistic.cdf(off + eps / (1 - pi))          # Qstar0 for every row (A set to 0)

def est_tmle(A, Y, pi, m0):
    q0 = _tmle_update(A, Y, pi, m0)
    psi = q0.mean()
    ic = (1 - A) / (1 - pi) * (Y - q0) + q0 - psi
    return psi, ic.std(ddof=1) / np.sqrt(len(Y))


def run_spec(dfs, spec, make_ml, seed):
    """All six E(Y^0) estimates for one dataset and one specification."""
    A = dfs['statin'].to_numpy()
    Y = dfs['Y'].to_numpy()
    untreated = dfs[dfs['statin'] == 0]
    is_para = spec in ('true', 'main')
    g_f, m_f = SPECS[spec]['g'], SPECS[spec]['m0']
    est = logit_est if is_para else make_ml

    np.random.seed(seed)
    # --- shared full-sample nuisances (2 fits) ---
    fb = None if is_para else logit_est
    pi = bound_p(fit_predict(g_f, dfs, dfs['statin'], dfs, est, fallback=fb), BOUND, 1 - BOUND)
    m0 = fit_predict(m_f, untreated, untreated['Y'], dfs, est, fallback=fb)

    out = {}
    for name, fn in [('gcomp', est_gcomp), ('ipw', est_ipw),
                     ('aipw', est_aipw), ('tmle', est_tmle)]:
        psi, se = fn(A, Y, pi, m0)
        out[f'{spec}_{name}_e0'], out[f'{spec}_{name}_se'] = psi, se

    # --- single cross-fit (5 folds, 1 partition): fold nuisances (10 fits) ---
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(dfs))
    folds = np.array_split(idx, 5)
    pi_cf = np.empty(len(dfs))
    m0_cf = np.empty(len(dfs))
    for i in range(5):
        train = dfs.iloc[folds[(i - 1) % 5]]              # nuisances from the previous fold
        train_untreated = train[train['statin'] == 0]
        evald = dfs.iloc[folds[i]]
        pi_cf[folds[i]] = fit_predict(g_f, train, train['statin'], evald, est, fallback=fb)
        m0_cf[folds[i]] = fit_predict(m_f, train_untreated, train_untreated['Y'], evald, est, fallback=fb)
    pi_cf = bound_p(pi_cf, BOUND, 1 - BOUND)

    # SC-AIPW: pooled pseudo-outcome mean; SE = sqrt(mean of within-fold IC variances / n)
    po = m0_cf + (1 - A) * (Y - m0_cf) / (1 - pi_cf)
    psi = po.mean()
    fold_vars = [po[f].var(ddof=1) for f in folds]
    out[f'{spec}_scaipw_e0'] = psi
    out[f'{spec}_scaipw_se'] = np.sqrt(np.mean(fold_vars) / len(Y))

    # SC-TMLE: targeting within each evaluation fold, pooled mean of updated predictions
    q0_all = np.empty(len(dfs))
    for f in folds:
        q0_all[f] = _tmle_update(A[f], Y[f], pi_cf[f], m0_cf[f])
    psi_t = q0_all.mean()
    ic = (1 - A) / (1 - pi_cf) * (Y - q0_all) + q0_all - psi_t
    fold_vars_t = [ic[f].var(ddof=1) for f in folds]
    out[f'{spec}_sctmle_e0'] = psi_t
    out[f'{spec}_sctmle_se'] = np.sqrt(np.mean(fold_vars_t) / len(Y))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=1000)
    ap.add_argument('--n', type=int, default=3000)
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
    df['ldl_160'] = np.where(df['ldl_log'] > np.log(160), 1, 0)
    df['age_30'] = df['age'] - 30
    df['age_30_sq'] = (df['age'] - 30) ** 2
    df['age_sqrt'] = np.sqrt(df['age'] - 39)
    df['risk_exp'] = np.exp(df['risk_score'] + 1)
    df['ldl_120'] = np.where(df['ldl_log'] > np.log(120), df['ldl_log'] ** 2, 0)

    t0 = time.time()
    for sid in range(args.start, args.end + 1):
        if sid in done_ids:
            continue
        dfs = df[df['sim_id'] == sid]
        if args.n < len(dfs):
            dfs = dfs.iloc[:args.n]
        dfs = dfs.copy().reset_index(drop=True)
        r = {'sim_id': sid}
        for spec in ['true', 'main', 'ml']:
            try:
                r.update(run_spec(dfs, spec, ml_estimator, BASE_SEED + sid))
            except Exception as e:
                print(f"  SPEC FAILURE sim_id {sid} spec {spec}: {type(e).__name__}: {e}",
                      flush=True)
                for m in ['gcomp', 'ipw', 'aipw', 'tmle', 'scaipw', 'sctmle']:
                    r[f'{spec}_{m}_e0'] = r[f'{spec}_{m}_se'] = np.nan
        rows.append(r)
        done = len(rows) - len(done_ids)
        el = time.time() - t0
        print(f"sim_id {sid}: ml aipw={r['ml_aipw_e0']:.4f} tmle={r['ml_tmle_e0']:.4f} "
              f"scaipw={r['ml_scaipw_e0']:.4f}   [{done} this run, {el/done:.1f}s/dataset]",
              flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    print(f"\nDone: {len(rows)} -> {args.out} in {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
