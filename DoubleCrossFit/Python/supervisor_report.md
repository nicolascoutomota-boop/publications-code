# Supervisor Report — Reproduction of Zivich & Breskin (Epidemiology 2021)

> **2026-07-21 — PLAN SUPERSEDED.** The compute problem below (§5) was resolved by agreeing a
> reduced 6-method design with Shaun (no HPC needed). The authoritative record of the current
> plan, decisions, correspondence, and open questions is now **`PROJECT_LOG.md`**. This report
> is kept as the record of the original reproduction findings (environment, software problems,
> verified facts), which all still stand. Results banked so far: 9 of the original 18 runs, all
> matching the paper's behaviour — see PROJECT_LOG.md §"Results banked so far".

**Student:** Nicolas · **Last updated:** 2026-07-21
**Paper:** Zivich PN, Breskin A. "Machine learning for causal inference: on the use of cross-fit
estimators." *Epidemiology* 2021;32(3):393–401 (arXiv:2004.10337).
**Code:** authors' public repo `pzivich/publications-code`, folder `DoubleCrossFit/Python/`,
run from my fork. **The authors' code is being run byte-for-byte unmodified** (verified with git).

Everything below is what happened while getting the reproduction running — findings, problems,
decisions, and open questions.

---

## 1. Status at a glance

- [x] Repo forked, cloned; authors' code verified **unmodified** against the published version
- [x] Simulated data generated with their `dgm.py` (2,000 datasets × n = 3,000; fixed seed)
- [x] Reproduction environment built (see §3) so their code runs **without touching a single line**
- [ ] Smoke test (`sim_single_example.py`, their own single-dataset example) — running
- [ ] The 18 production runs (6 estimators × 3 setups) — **blocked in part by compute, see §5**
- [ ] Comparison against the paper's figures/tables

## 2. Verified facts about their simulation (from the code itself)

| Parameter | Value in code | Note |
|---|---|---|
| Sample size per dataset | **n = 3,000** (`dgm.py`) | ⚠️ You recalled n = 500 in our meeting — the paper and code both use 3,000 |
| Number of simulated datasets | 2,000 (`dgm.py`, `sims = 2000`) | |
| True ATE (risk difference) | −0.1081508 (hard-coded in every sim script) | |
| RNG seed for data generation | 1015033030 (`dgm.py`) | Data generation is exactly reproducible |
| Estimator randomness | **Not seeded** in the sim scripts | Split/RF/NN randomness varies run to run → estimator results reproduce statistically, not bit-for-bit |
| Setups per estimator | 1 = correct parametric, 2 = main-terms parametric, 3 = super learner | |
| G-formula variance | 250 bootstrap resamples per dataset | |
| Cross-fit estimators | 100 different sample splits per dataset, median-combined | |

## 3. Software problems found (documented, NOT fixed in their code)

Their code is from 2019–2020; on 2026 library versions it crashes in three independent places.
Per the project rule, **nothing was changed in their files** — instead the environment was pinned
to era-appropriate versions where the original code runs as-is:

| # | Problem | Where | Modern behaviour | Resolution (environment, not code) |
|---|---|---|---|---|
| 1 | `LogisticRegression(penalty='none')` removed | all sim scripts + super_learner.py | crash on scikit-learn ≥ 1.4 | pin scikit-learn 1.1.3 (string form fully supported) |
| 2 | statsmodels link passed as **class**, not instance | estimators.py L224–226 (`IPTW.fit`) | `TypeError` on statsmodels ≥ 0.13.0 | pin statsmodels 0.12.2 |
| 3 | positional `[0]` on a labelled pandas Series | estimators.py L611 (`TMLE.fit`) | `KeyError` on pandas ≥ 3.0 | pin pandas 1.5.3 |
| 4 | zepid imports `pkg_resources` | zepid package | gone from new setuptools | pin setuptools < 81 |

**Pinned reproduction environment** (`.venv-pinned`): Python 3.9.7, numpy 1.22.4, scipy 1.7.3,
pandas 1.5.3, scikit-learn 1.1.3, statsmodels 0.12.2, pygam **0.8.0 (authors' exact version)**,
zepid 0.9.1, matplotlib 3.5.3. Authors' README lists their originals as numpy 1.17.2, pandas
0.24.2, sklearn 0.22.1, pygam 0.8.0, zepid 0.8.1 — those are too old to install on current
macOS/Python, so this is the closest stack on which their code runs unmodified.

*Note for full transparency:* before this rule was established, a setup guide had me apply the
`penalty='none'`→`penalty=None` find-replace inside their files. This was **reverted**; git
confirms their files are now identical to the published repo, and the pinned environment makes
the edit unnecessary.

## 4. A discrepancy to be aware of

- **Library versions differ from the authors' originals** (nothing older installs on a 2026 Mac).
  Numerical differences from library evolution (e.g., sklearn 0.22 → 1.1 random forest/MLP
  internals) are possible; Monte Carlo–level agreement with the paper is the realistic target,
  not bit-identical numbers. This matches your "plus or minus a little bit of error" guidance.
- **The estimator scripts set no random seed** (only data generation is seeded), so even the
  authors could not exactly regenerate their own tables — another reason exact matching is the
  wrong bar.

## 5. ⚠️ The main practical problem: compute

I timed **every estimator × setup on real data on my laptop** (4-core i7 MacBook Pro) and
extrapolated ×2,000 datasets:

| Script | Setup 1 | Setup 2 | Setup 3 (super learner) |
|---|---|---|---|
| sim_iptw.py | ~40 min | ~40 min | ~1 day |
| sim_aipw.py | ~30 min | ~30 min | ~2 days |
| sim_tmle.py | ~1 h | ~1 h | ~2 days |
| sim_gform.py (250 bootstraps) | ~17 h | ~17 h | **~209 days** |
| sim_dcaipw.py (100 splits) | ~4 days | ~4 days | **~340 days** |
| sim_dctmle.py (100 splits) | ~4 days | ~4 days | **~327 days** |

(single-process times; parallelising ~6× on my laptop divides these by ~6)

**Update after switching to the pinned (era-appropriate) environment:** parametric fits are
~10× faster there (e.g. full AIPW/TMLE setup-1 runs finished in ~4 min each), so the feasible
part shrinks well below the original 4–6-day estimate. The super learner, however, is *not*
faster (measured ≈ 25 s per fit at K=5 on n=3,000; production uses K=10 ≈ ~50 s), so the
conclusion for the three heavy runs is unchanged.

**Conclusion:**
- 15 of the 18 runs are feasible on my laptop: **~4–6 days of continuous running** (likely
  less — parametric parts measured ~10× faster on the pinned environment).
- The 3 super-learner-heavy runs (g-formula/DC-AIPW/DC-TMLE at setup 3) total ≈ **21,000
  core-hours ≈ 5 months on my laptop — not feasible.** The authors' README itself says these
  must be "broken into pieces to run in parallel"; they evidently used a cluster.

**Questions for you:**
1. Can I get access to university HPC (UiO Educloud/Fox)? ~128 cores would finish the three heavy
   runs in about a week.
2. Alternatively, do you want a reduced setting for those three (e.g., your suggestion of 1,000
   datasets, or fewer sample-splits) — accepting it deviates from the paper's settings?
3. Is "everything except the three heavy ML runs" a useful first deliverable for our next meeting?

## 6. Minor notes

- The g-formula script is the only one using bootstrap variance (250 refits/dataset) — that's why
  its setup 3 explodes: 251 super-learner fits per dataset.
- `sim_dcaipw.py`/`sim_dctmle.py` wrap each dataset in `try/except` and record NaN on failure —
  the output CSVs may contain missing rows by design; worth remembering when summarising.
- Full run log with timings and library details: `notes.md` in this folder.
