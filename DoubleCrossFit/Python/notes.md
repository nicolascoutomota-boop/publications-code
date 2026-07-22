# Reproduction lab notebook — Zivich & Breskin cross-fit estimators

Paper: Zivich PN, Breskin A. "Machine learning for causal inference: on the use of
cross-fit estimators." *Epidemiology* 2021;32(3):393-401. (arXiv:2004.10337)
Local copy: `../zivich-breskin-2021-crossfit-estimators-arxiv.pdf`

Goal (Phase 1): reproduce the published results with the authors' code **run exactly as
published — zero modifications**, keep every output file, then meet the supervisor before
any variations.

**PROJECT RULE: never edit the authors' files.** Any problem is documented here and in
`supervisor_report.md`, and solved by pinning the *environment*, never by changing code.
(Early on, a setup guide had me find-replace `penalty='none'` → `penalty=None` inside
their files; this was **reverted on 2026-07-19** — `git status` confirms all authors'
files are byte-identical to the published repo.)

## Environments

- `.venv` — first attempt, modern packages (Python 3.12, numpy 2.5, pandas 3.0,
  sklearn 1.9). Authors' code CRASHES on it in 3 places (see Issues). Kept for reference;
  do not use for production runs.
- **`.venv-pinned` — the reproduction environment (USE THIS).** Python 3.9.7 with
  era-appropriate versions so the unmodified code runs:
  numpy 1.22.4, scipy 1.7.3, pandas 1.5.3, scikit-learn 1.1.3, statsmodels 0.12.2,
  pygam 0.8.0 (authors' exact version), zepid 0.9.1, patsy 1.0.2, matplotlib 3.5.3,
  setuptools < 81.
  Run scripts with: `.venv-pinned/bin/python sim_xxx.py`
  (authors' own versions per repo README — numpy 1.17.2, pandas 0.24.2, sklearn 0.22.1,
  pygam 0.8.0, zepid 0.8.1 — no longer install on a 2026 Mac; this is the closest
  working stack.)

## ISSUES FOUND — reported to supervisor (code NOT modified)

1. **scikit-learn ≥ 1.4 removed `penalty='none'`** — used throughout the sim scripts and
   super_learner.py. Works as-is on pinned sklearn 1.1.3 (no deprecation there).
2. **statsmodels ≥ 0.13.0 rejects link classes** — `estimators.py` L224–226 passes
   `sm.families.links.identity` (a class) into `Binomial(...)`; modern statsmodels raises
   `TypeError`, breaking `IPTW.fit()` in every setup. Works on pinned statsmodels 0.12.2.
   (Tested: 0.14.6 ✗, 0.13.5 ✗, 0.13.2 ✗, 0.12.2 ✓.)
3. **pandas 3.x removed positional `[0]` on labelled Series** — `estimators.py` L611
   (`self._epsilon[0]` in `TMLE.fit()`) raises `KeyError: 0`, breaking TMLE in every
   setup. Works on pinned pandas 1.5.3.
4. **zepid imports `pkg_resources`** — needs `setuptools<81` in the venv.
5. Dependency chain forced by (2): statsmodels 0.12.2 needs scipy < 1.8 (`_centered`
   import) and numpy < 1.24 (`np.MachAr`), hence scipy 1.7.3 + numpy 1.22.4 + Python 3.9.

## Key parameters (verified in code)

- `dgm.py`: n = 3000 per dataset, sims = 2000 datasets, seed 1015033030.
  NOTE for supervisor: the paper's sample size is 3,000, not the 500 he recalled.
- `truth = -0.1081508` (true risk difference / ATE), hard-coded in every sim script.
- `setup` variable in each sim script: 1 = correct parametric, 2 = main-terms
  (misspecified), 3 = machine learning / super learner.
- Estimator randomness (sample splits, RF, neural net) is NOT seeded in the sim scripts —
  results reproduce statistically, not bit-for-bit.

## Measured runtimes (this MacBook, 4-core i7; per-dataset × 2,000 extrapolation)

| Script | Setup 1 | Setup 2 | Setup 3 (SL) |
|---|---|---|---|
| sim_iptw.py | ~40 min | ~40 min | ~1 day |
| sim_aipw.py | ~30 min | ~30 min | ~2 days |
| sim_tmle.py | ~1 h | ~1 h | ~2 days |
| sim_gform.py | ~17 h | ~17 h | ~209 days ⚠️ |
| sim_dcaipw.py | ~4 days | ~4 days | ~340 days ⚠️ |
| sim_dctmle.py | ~4 days | ~4 days | ~327 days ⚠️ |

Feasible locally (15 runs): ~4–6 days wall-clock parallelised. The three ⚠️ runs need
HPC (≈21,000 core-hours) — supervisor question. Timings measured on the modern env;
pinned-env speeds to be confirmed (same order of magnitude expected).

## Run log

| Date | Script | Setup | Env | Notes / runtime |
|------|--------|-------|-----|-----------------|
| 2026-07-19 | dgm.py | — | .venv | Generated statin_sim_data.csv (6M rows, ~346 MB). |
| 2026-07-19 | sim_single_example.py | — | .venv | First smoke test (with penalty edits, since reverted). |
| 2026-07-19 | sim_single_example.py | — | .venv-pinned | **PASSED — all 11 estimator blocks, ~6.5h total, exit 0.** Single-dataset RDs (truth −0.108): gform −0.14/−0.09 (param/SL), IPTW −0.13/−0.12, AIPW ≈−0.11/−0.12, TMLE −0.12/−0.12, DC-AIPW −0.09/≈−0.12, DC-TMLE ≈−0.12/−0.11. Output: smoke_test_pinned_output.txt |
| 2026-07-19 | run_sim_gform_setup2.py | 2 | .venv-pinned | Launched (copy script) after smoke completion freed a slot. |
| 2026-07-20 | sim_gform.py | 1 | .venv-pinned | **DONE ~30h** (250 bootstraps x 2000 datasets). Bias 0.0002, ASE 0.0166 vs ESE 0.0173, coverage 0.9355. Sanity check PASS. → gform_results1.csv |
| 2026-07-20 | run_sim_aipw_setup3.py | 3 | .venv-pinned | Launched in the slot freed by gform setup 1. |
| 2026-07-21 | run_sim_iptw_setup3.py | 3 | .venv-pinned | **DONE ~2 days**. Bias 0.0107, ASE 0.0227, coverage 0.945. Sanity PASS. → iptw_results3.csv (result #9). |
| 2026-07-21 | (session closed) | — | — | **All 5 background jobs died when the previous session ended.** iptw3 and gform2 had already finished (saved). AIPW setup 3 (~1d in), DC-AIPW setup 1 (~2d in), DC-TMLE setup 1 (~2d in) died with NO output — these write only at the end, so that compute was lost. See PROJECT_LOG.md. |
| 2026-07-21 | timing benchmark | — | .venv-pinned | Machine idle → clean measure: one super-learner fit (K=10, n=3000) = **13 s**. Basis for reduced-design timing table in PROJECT_LOG.md. |

## PLAN CHANGE (2026-07-21): reduced design agreed with supervisor

The plan has moved from the full 18-run reproduction to Shaun's reduced 6-method design.
Full details + open questions in **PROJECT_LOG.md** (single source of truth for decisions).
Short version: keep IPW / AIPW / TMLE; g-computation without bootstrap (CI from empirical SE);
replace double cross-fit with simple 5-fold cross-fit AIPW & TMLE; drop DC entirely; keep the
super-learner (incl. neural net + full random forest) as published; 1,000 datasets; seed our
own code. Awaiting Shaun's choice of cross-fit repetition count before implementing.
| 2026-07-19 | sim_aipw.py | 1 | .venv-pinned | **DONE ~4 min** (far faster than modern-env estimate). Bias 0.0001, ASE 0.0198 vs ESE 0.0206, coverage 0.939. Sanity check PASS. → aipw_results1.csv (2000 rows) |
| 2026-07-19 | sim_tmle.py | 1 | .venv-pinned | **DONE ~4 min**. Bias 0.0003, ASE 0.0195 vs ESE 0.0205, coverage 0.9355. Sanity check PASS. → tmle_results1.csv (2000 rows) |
| 2026-07-19 | sim_iptw.py | 1 | .venv-pinned | **DONE ~37 min** (GEE fits dominate). Bias 0.0067, ASE 0.0247 vs ESE 0.0237, coverage 0.948. Sanity check PASS. → iptw_results1.csv (2000 rows) |
| 2026-07-19 | run_sim_iptw_setup2.py | 2 | .venv-pinned | **DONE ~40 min**. Bias −0.0224, ASE 0.0231 vs ESE 0.0229, coverage 0.8655 — misspecification bias, consistent with paper. → iptw_results2.csv (2000 rows) |
| 2026-07-19 | sim_dcaipw.py | 1 | .venv-pinned | Launched (authors' file as published; ~28h est. after DC-AIPW parametric smoke block passed in 51s/dataset). |
| 2026-07-19 | run_sim_iptw_setup3.py | 3 | .venv-pinned | Launched (copy script; SL path validated by smoke test; ~1 day est.). |
| 2026-07-19 | sim_dctmle.py | 1 | .venv-pinned | Launched (authors' file as published; ~1.7d est. after DC-TMLE parametric smoke block passed, 1.2 min/dataset). |

Smoke-test SL timings (K=5, n=3000, one dataset): gform-SL block 104.8 min (251 fits ≈ 25 s/fit);
DC-AIPW-SL block 173.2 min (100 splits); DC-TMLE parametric 1.2 min. Production setup-3 uses
K=10 ≈ 2× ⇒ dcaipw/dctmle setup-3 ≈ 5–6 h/dataset ⇒ ~480 core-days each. CLUSTER CONFIRMED
NECESSARY for the three heavy runs; pinned env does not change that conclusion.
| 2026-07-19 | run_sim_aipw_setup2.py | 2 | .venv-pinned | **DONE ~4 min**. Bias −0.0162, ASE 0.0195 vs ESE 0.0200, coverage 0.844 — bias + undercoverage under misspecification, as the paper predicts. → aipw_results2.csv (2000 rows) |
| 2026-07-19 | run_sim_tmle_setup2.py | 2 | .venv-pinned | **DONE ~4 min**. Bias −0.0173, ASE 0.0192 vs ESE 0.0182, coverage 0.849 — bias + undercoverage under misspecification, mirrors AIPW setup 2. → tmle_results2.csv (2000 rows) |

Setup-2/3 mechanism (approved): untracked copies `run_sim_<est>_setup<2|3>.py`, byte-identical
to the authors' scripts except the one `setup =` digit; proof in `setup_copies_diffproof.txt`.
Authors' files remain pristine throughout.
Note: parametric runs are ~10× faster on the pinned env than my modern-env benchmarks —
runtime table above is pessimistic for setups 1–2; SL-based estimates to be recalibrated
from the smoke test's timing blocks.

<!-- Add one row per run: date, script, setup number, env, runtime, anything strange. -->
