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

### OPEN QUESTIONS awaiting Shaun's reply (asked 2026-07-21)
- **How many cross-fit repetitions (k)?** zepid's `SingleCrossfit*` default is
  `n_partitions=100`, which would blow the budget back up to ~1,000 fits/dataset. Shaun's
  "10 fits" assumes k=1. I suggested a few (3–5) for stability; awaiting his choice.

### MY STANDING TO-DOs once k is fixed
- [ ] Time one full dataset under the exact agreed settings before promising a schedule.
- [ ] **Set a seed in my own driver code** so our results are reproducible (theirs are not).
- [ ] Implement the 6-method driver. Use zepid's `SingleCrossfitAIPTW` / `SingleCrossfitTMLE`
      (Zivich's own package — no need to write cross-fit code ourselves; `fit(n_splits=5,
      n_partitions=k, random_state=…)`).
- [ ] For comparability, store **two CIs per method**: one from each method's own SE, one
      from the empirical SE. (Flagged to Shaun: g-comp otherwise uses the true SE while
      AIPW/TMLE use estimated SEs, so coverage isn't like-for-like. Costs no extra compute.)
- [ ] Store all point estimates for every dataset (Shaun asked explicitly).

---

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
