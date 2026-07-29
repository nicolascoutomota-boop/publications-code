# Project Log — Zivich & Breskin reproduction

Single source of truth for **what was decided, when, and why**. Newest first.
Supervisor = Shaun. Student = Nicolas. Paper = Zivich & Breskin, *Epidemiology* 2021
(arXiv:2004.10337). Code = `pzivich/publications-code`, folder `DoubleCrossFit/Python/`.

Governing rule for the whole project: **never modify the authors' code.** Problems are
documented and the *environment* is adapted instead. Their 9 source files are verified
pristine against the published repo (`git status` clean on them).

---

## The agreed plan (as of 2026-07-21)

Shaun proposed a **reduced design** that answers the same scientific question without the
massive compute. His email + my reply (sent 2026-07-21) settled the following. **Six methods**,
each applied to every simulated dataset, storing the ATE point estimate for each:

1. **IPW** — unchanged.
2. **G-computation, no bootstrap.** Point estimate only (1 super-learner fit/dataset instead
   of 251). No per-dataset SE. Build CIs using the **empirical ("true") SE** = the standard
   deviation of the point estimates across all datasets, in place of the bootstrap SE.
3. **AIPW, no cross-fitting** — unchanged.
4. **TMLE, no cross-fitting** — unchanged.
5. **AIPW with simple 5-fold cross-fitting** — replaces DC-AIPW.
6. **TMLE with simple 5-fold cross-fitting** — replaces DC-TMLE.

**Dropped:** double cross-fitting (DC-AIPW, DC-TMLE). Shaun: "not commonly used… I am
surprised Zivich & Breskin used it."

**Super-learner:** left exactly as published — **keep the neural net, keep the 500-tree
random forest.** (Do NOT do the optional neural-net-removal extension yet.)

**Datasets:** 1,000 is enough (Shaun agreed; Monte-Carlo error on coverage ≈ 0.7pp at 1,000
vs 0.5pp at 2,000 — immaterial).

**Cost of the reduced design** (measured 2026-07-21, one SL fit K=10 n=3000 = 13 s idle):
fits/dataset = 6 + 20k, where k = number of cross-fit repetitions.
| k (repetitions) | fits/dataset | % of original 1,456 | 1,000 sets | 2,000 sets |
|---|---|---|---|---|
| 1 | 26 | 1.8% | ~1 day | ~2 days |
| 3 | 66 | 4.5% | ~2.5 days | ~5 days |
| 5 | 106 | 7.3% | ~4 days | ~8 days |
(4 cores; pad ~1.5× for real-run overhead/thermal throttling.)

### RESOLVED (Shaun, 2026-07-21, second reply)
- **Cross-fit repetitions k = 1** (a single split). Shaun: repeated cross-fitting is a real
  technique that lowers the ATE estimator's SE and improves single-dataset reproducibility,
  but he cares about *summary statistics over 1,000–2,000 datasets*, not per-dataset SE or
  single-dataset reproducibility — so one split is fine, and cheaper. ⇒ `n_partitions=1`.
- **1,000 datasets** (confirmed "entirely satisfactory").
- ⇒ Design is now **fully specified**: 26 super-learner fits/dataset, ~1 day on my laptop.
  (Note for interpretation: with a single split the two cross-fit estimators carry extra
  split-induced variance, which will slightly inflate their empirical SD vs a repeated-split
  version. That is a genuine property of single-split cross-fitting, not an error — Shaun
  made this trade-off knowingly.)

### IMPLEMENTATION status
- [x] **Driver built and smoke-tested:** `reduced_design.py` (NEW file; does not touch the
      authors' code). Runs all six methods per dataset, writes crash-safe after each dataset,
      chunkable via `--start/--end` for parallel cores. Smoke test (2 datasets, K=2) passed —
      all six give sane estimates near the truth; zepid single-crossfit works at
      `n_splits=5, n_partitions=1`.
- [x] **Seeded** per dataset (`BASE_SEED + sim_id`) so the run is reproducible and chunkable.
- [x] Uses zepid's `SingleCrossfitAIPTW` / `SingleCrossfitTMLE` (Zivich's own package).
- [x] Stores all six point estimates + SEs (where available) for every dataset.
- [x] **Full run LAUNCHED** 2026-07-21: 1,000 datasets, K=10, as **2 chunks × RF n_jobs=2**
      (`reduced_chunkA.log` datasets 1–500, `reduced_chunkB.log` 501–1000). Resume-enabled
      (re-run same command to continue after any interruption). **ETA ~3.4 days.**
      - Runtime tuning (measured on this 4-core i7-8557U laptop, which throttles under load):
        4 procs×1 thread = 469 s/dataset (~5.4 d); 1 proc×4 threads = 386 s (~4.5 d);
        **2 procs×2 threads = 297 s/dataset (~3.4 d) — chosen.**
      - `--rf-jobs` is a pure speed flag (parallel tree building); it does not change the
        statistical method. The authors' `super_learner.py` is untouched; n_jobs is set on the
        estimator object at runtime.
      - Earlier "~1 day" estimate was wrong: based on an over-optimistic 13 s/fit idle
        measurement; real sustained cost is ~45–70 s/fit on this throttling laptop.
- [x] **RUN COMPLETE 2026-07-26: all 1,000 datasets, 0 NaNs.** Merged to
      `reduced_results_all.csv`; summary in `reduced_summary.csv` (`summarize_reduced.py`).

### RESULTS (1,000 datasets, truth = -0.1081508)
| method | bias | ESE | ASE | cov(own SE) | cov(true SE) |
|---|---|---|---|---|---|
| IPW | +0.0115 | 0.0210 | 0.0227 | 0.937 | 0.915 |
| **G-computation (no bootstrap)** | **+0.0264** | 0.0167 | — | — | **0.640** |
| AIPW | +0.0043 | 0.0192 | 0.0166 | 0.908 | 0.948 |
| TMLE | -0.0014 | 0.0201 | 0.0165 | 0.901 | 0.952 |
| AIPW 5-fold cross-fit | -0.0029 | 0.0235 | 0.0221 | 0.935 | 0.952 |
| TMLE 5-fold cross-fit | +0.0008 | 0.0205 | 0.0205 | 0.938 | 0.956 |

**Reproduces the paper's message.** With super-learner nuisance estimation, plug-in
**g-computation is the most biased (+0.026) and its CIs cover only 64%** even using the true
SE — a pure bias effect. The doubly-robust estimators (AIPW, TMLE) are near-unbiased, and the
**cross-fit versions are best**: smallest bias and ~95% coverage both ways. Also visible:
AIPW/TMLE *without* cross-fitting show own-SE coverage ~0.90 (slightly low, from in-sample
nuisance fitting), which cross-fitting lifts back to ~0.95.

---

### 2026-07-28 — Shaun's post-results email + Table-3 deliverable
- Deadline context: end-of-internship presentations 11 August.
- Shaun listed 4 possible next runs (RF-only; two separate SLs for E[Y|X=1,Z]/E[Y|X=0,Z];
  n=1500; a variant of #2) — explicitly said DON'T run yet; he'll prioritise.
- Asked us to reproduce Z&B Table 3 (verified structure, paper p.9: 6 estimators x 3 specs,
  cols Bias/RMSE/ASE/ESE/CLD/Coverage) with his two changes (g-comp empirical-SE CIs;
  single instead of double cross-fit), plus later E(Y^1)/E(Y^0) versions (needs re-run —
  nothing stored the arms; zepid SC classes only report the difference).
- Timings measured for his questions: RF-only fit = 0.85s vs SL 17.8s (~1/20 → few hours);
  n=1500 SL fit = 0.67x (→ 2-3 days); #2 est. +1/3 to +1/2 vs current.
- **Important verified finding:** zepid SingleCrossfit trains each nuisance on ONE split
  (n/5 = 600 obs), not the complementary 4/5 (textbook k-fold). Evaluation split i uses
  models from split i-1 (cyclic). Same fit count, valid cross-fitting, but less training
  data per nuisance → likely explains SC methods' larger ESE. Flagged to Shaun; textbook
  version would need our own driver code.
- **Table-3 completion run launched:** `sc_parametric.py` — SC-AIPW/SC-TMLE with parametric
  nuisances, True + Main-effects specs, sim_id 1-1000, seeded, resume-safe (~80 min).
  `build_table3.py` assembles the full 3-panel table → `table3_reproduction.csv`.

### 2026-07-29 (later) — Shaun's confirmations + presentation spec + STUDY 3 LAUNCHED
- **Meeting: Friday 4pm UK.**
- **Presentation (11 Aug): 20 min total = 15 talk + 5 questions.** Audience: Biostatistics
  Unit statisticians, mixed experience, many new to debiased ML. Suggested structure:
  first half = essentials of debiased ML for ATE (why g-comp with data-adaptive nuisances
  fails, why AIPW/TMLE are better), not too complicated; second half = why this simulation
  work + results + 1-2 conclusions. Content to be discussed Friday.
- Approvals: arms for g-computation only ✓; n=1500 via first 1,500 rows ✓; same datasets
  for other studies where possible ✓ (we use sim_id 1-1000 throughout).
- Table 3 sent to him (email_table3_draft.md).
- **Study 3 running:** `study3_twomodel.py`, 2 detached chunks (ppid=1, caffeinate -ims),
  ~45 s/dataset → ~6 h. Two-model g-computation; arm-specific formulas; 'true' spec
  treated-arm includes ldl_130 (the interaction term active only under treatment in Z&B's
  single-model formula). Records ey1/ey0/ate per spec per dataset.
  NOTE: first launch attempt silently failed (zsh word-splitting in a loop) — caught by
  verification; relaunched explicitly and confirmed computing.

### 2026-07-29 — NEW PLAN from Shaun: g-computation only, three studies, arms tables
- Rationale: time is short (presentations 11 Aug); the finished study already demonstrated
  the DR+cross-fit fix, so he'll assume it works from here and focus on **g-computation's
  failure modes**. He cares more about **E(Y^1) and E(Y^0) separately** than the ATE
  (will explain why at the meeting).
- Confirmed to him: stored results have no arms → g-comp-only re-runs will record them
  (cheap); five-method ML arms would need the multi-day re-run (assumed not wanted).
- **The three studies (g-computation ONLY, each → Table-3-format tables for ATE, E(Y^1),
  E(Y^0); empirical-SE CIs):**
  1. RF replacing the super learner (n=3000) — ~under 1 h compute
  2. n=1500 (plan: first 1,500 rows of each existing dataset; flagged to him) — ~2-3 h
  3. **START HERE (his priority):** two separate super learners for E(Y|X=1,Z) (fit on
     treated) and E(Y|X=0,Z) (fit on untreated) — needs our own small driver (their
     GFormula fits a single model with treatment as covariate) — ~half a day
- Meeting to be scheduled (he'll explain the "why" + the algebra-based variant of #3).

## Correspondence & decisions (chronological)

### 2026-07-21 — Shaun's reduced-design proposal + my reply
- Shaun proposed the 6-method reduced design above; asked (a) confirm no exact reproduction
  is possible because estimation isn't seeded, (b) how long 1,000 / 2,000 datasets take,
  (c) any problems.
- My reply: (a) confirmed — data is seeded, estimation isn't (RF + NN random), so ML results
  aren't bit-reproducible by anyone; I'll seed our own code. (b) timing table above. (c)
  raised the `n_partitions` multiplier as the one thing to decide; noted zepid already has the
  single-crossfit estimators.

### 2026-07-20/21 — Q&A on the compute breakdown
- Shaun asked which learners are slow and whether the load is "very largely" double-crossfit.
- Measured: **random forest ≈ 80% of super-learner runtime** at every n from 250–12,000; neural
  net small and erratic (<1%–10%); removing the NN saves ~a tenth, so its case is
  methodological not computational. RF cost is nearly flat in n (500 trees is a fixed cost),
  so **smaller sample sizes help less than expected** (n=250 only ~2× faster than n=3,000).
- Double cross-fitting ≈ three-quarters of total setup-3 cost; g-computation ≈ a fifth
  (its 250-replicate bootstrap); IPW/AIPW/TMLE < 1%. Verified by timing the authors' own
  `sim_single_example.py` (one dataset ≈ 8 h; ×2,000 gives the month-scale figures).

### 2026-07-19/20 — first compute email
- Reported to Shaun that the 3 super-learner-heavy runs (g-form/DC-AIPW/DC-TMLE at setup 3)
  need ~months on a laptop; asked about compute options. Shaun declined HPC ("shouldn't need
  that much computing… don't want to fry the planet") and produced the reduced design instead.

### Supervisor's original standing instructions (from action-report / meetings 48–50)
- **Reproduce first, then talk.** (Now superseded into the reduced design by mutual agreement.)
- Success = reproduction + full results (all estimators, all metrics — the authors only
  published a subset) + a few sample sizes (n = 250, 1,000, 2,000) + understanding.
- Binary treatments only; do not pursue continuous treatments.
- Keep every output file — the "full results" deliverable is built from them.
- Optional/holiday extensions (NOT started, by instruction): remove the neural net; reduce to
  1,000 datasets; port to R.

---

## Results banked so far (published settings, pinned env)

These are from the **original** 18-run reproduction attempt. They remain valid as a reference
check that the environment reproduces the paper's behaviour, even though the plan has since
moved to the reduced design. Truth = −0.1081508.

| # | Estimator | Setup | Bias | ASE | ESE | Coverage | Reads as |
|---|---|---|---|---|---|---|---|
| 1 | AIPW | 1 (correct) | 0.0001 | 0.0198 | 0.0206 | 0.939 | unbiased, ~95% ✓ |
| 2 | TMLE | 1 | 0.0003 | 0.0195 | 0.0205 | 0.936 | unbiased, ~95% ✓ |
| 3 | IPW | 1 | 0.0067 | 0.0247 | 0.0237 | 0.948 | unbiased, ~95% ✓ |
| 4 | AIPW | 2 (main-terms) | −0.0162 | 0.0195 | 0.0200 | 0.844 | bias + undercoverage ✓ |
| 5 | TMLE | 2 | −0.0173 | 0.0192 | 0.0182 | 0.849 | bias + undercoverage ✓ |
| 6 | IPW | 2 | −0.0224 | 0.0231 | 0.0229 | 0.866 | misspecification bias ✓ |
| 7 | G-comp | 1 | 0.0002 | 0.0166 | 0.0173 | 0.936 | unbiased, ~95% ✓ |
| 8 | G-comp | 2 | −0.0226 | 0.0171 | 0.0176 | 0.722 | strong undercoverage ✓ |
| 9 | IPW | 3 (super learner) | 0.0107 | 0.0227 | — | 0.945 | ~95% ✓ |

All nine behave exactly as the paper describes. CSVs in this folder (`*_results{1,2,3}.csv`),
2,000 rows each.

### Runs LOST when the previous session closed (2026-07-21)
The five background jobs died with the session; those that hadn't finished write output only
at the end, so their compute was lost:
- AIPW setup 3 — died ~1 day in (no output). **Needed under the reduced plan — to rerun.**
- DC-AIPW setup 1 — died ~2 days in. **Dropped from the plan — not rerunning unless the full
  original reproduction is still wanted as a reference.**
- DC-TMLE setup 1 — died ~2 days in. **Dropped — same as above.**
- (IPW setup 3 finished first → result #9 above, saved.)
- (g-formula setup 2 finished → result #8, saved.)

---

## Deliverables produced (in this folder unless noted)
- `notes.md` — detailed run-by-run lab log with timings.
- `supervisor_report.md` — findings/problems/environment writeup for Shaun.
- `code_reference.pdf` (in `Project/`) — 29-page precise explanation of every source file.
- `setup_copies_diffproof.txt` — proof the `run_sim_*_setup{2,3}.py` copies differ from the
  authors' files by only the one `setup=` digit.
- `email_draft.md`, `email_reply_draft.md`, `email_reply2_draft.md` — supervisor emails.
- `.gitignore` — excludes `.venv*/`, `statin_sim_data.csv` (346 MB), `__pycache__/`.
