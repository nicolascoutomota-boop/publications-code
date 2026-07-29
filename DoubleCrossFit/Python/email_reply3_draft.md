# Reply to Shaun — timing, Table 3, arms, and the cross-fit structure flag

VERIFIED 2026-07-28 (idle machine) before sending:
- Z&B Table 3 (paper p.9): 6 estimators x 3 specs (True / Main-effects / Machine learning),
  columns Bias, RMSE, ASE, ESE, CLD, Coverage. Our finished run = ML panel only. True/Main
  rows for gform/ipw/aipw/tmle exist from the earlier reproduction (2,000 datasets, will
  subset to sim_id 1-1000 for comparability); SC-AIPW/SC-TMLE parametric rows NOT yet run
  (cheap: logistic nuisances, ~1-2h).
- zepid crossfit.py: nuisances are fit PER SPLIT (each on ~n/5=600 obs), evaluation split i
  uses models from split i-1 (cyclic). NOT textbook 4/5-training 5-fold. Fit count same.
- SingleCrossfit classes expose only the difference (risk_difference / ace) — no arm-level
  E[Y^1], E[Y^0] attributes -> arms need re-run + small driver extension.
- RF-only fit = 0.85s vs SL fit 17.8s (n=3000, K=10) -> #1 is ~5% of cost, few hours.
- SL fit n=1500 = 11.9s = 0.67x -> #3 ~ 2-3 days.
- Timing story: 1-day estimate assumed ~4x parallel scaling; measured effective throughput
  gave ~3x less (thermal throttling; best config was 2 procs x 2 RF threads = 297s/dataset);
  plus ~1 day idle from session deaths + Mac sleep. Computing ~3 days, calendar ~4.5.

---

Subject: Re: Reduced run finished

Dear Shaun,

On the career talks — yes, I am being invited, and I've already attended some of them.
Since the presentations are on 11 August: could I ask what's expected of them? What I should
present, the format and length, any guidelines, and who the audience will be.

**What I expected the run to take.** I estimated about a day, and the gap is on me. Two
things: my estimate assumed the laptop's four cores would give a near four-fold parallel
speed-up, but under days of sustained load it throttles and delivered closer to 1.3x; and
about a day was lost to interruptions (the laptop going to sleep, my terminal closing)
before I'd made the run fully robust to them. The actual computing was about three days,
plus roughly a day idle.

**Your timing questions, now measured properly:**

- Number 1 (random forest only): much faster, not almost as long — a plain random forest
  fit is about a twentieth of a super-learner fit, so the whole run would be a few hours,
  well under a day. The "almost as long" was a misunderstanding of something I said earlier:
  the ~80% figure was the forest's share of cost inside the super learner, where it is refit
  eleven times per super-learner fit. On its own it is fit once, so it's cheap.
- Number 3 (n = 1500): roughly two-thirds of the current run, so 2-3 days — not half,
  because the random forest's cost falls only gently with sample size.
- Number 2 (two separate super learners): I'd estimate a third to a half longer than the
  current run — the outcome-model fits double, but each is on a subset.
- And on your thought about focusing on g-computation only: that would be very cheap — one
  super-learner fit per dataset instead of twenty-six, so a few hours per scenario. It would
  make trying several scenarios (your 1-4) quick.

**Table 3: yes.** Three parts, honestly labelled: the machine-learning rows come directly
from the run just finished. The "true" and "main-effects" rows for g-computation, IPW, AIPW
and TMLE come from my earlier runs of Z&B's unmodified code, which stored all per-dataset
output. The single cross-fit rows for those two parametric specifications haven't been run
yet, but with parametric nuisance models they are cheap — I'll run them now so the complete
table (all three specifications, your two changes applied) is ready for the meeting. For
comparability I'll compute everything on the same 1,000 datasets.

**E(Y^1) and E(Y^0) separately: possible, but needs a re-run.** All runs so far stored only
the difference, and for the cross-fit methods Zivich's package only reports the difference,
so I'd add a small extension to record the two arms. For the machine-learning panel that
re-run costs the same as what we just did (~3-4 days), so it would be worth folding into
whichever of your points 1-4 we prioritise, to pay for the computing only once.

**One technical point I found while double-checking the package code.** Zivich's single
cross-fit is not the textbook five-fold procedure: with five splits, the nuisance models
used for each fifth are trained on one other fifth (about 600 people), not on the remaining
four-fifths. The number of super-learner fits is the same, and it is still valid
cross-fitting, but each nuisance model sees less data than in the standard scheme — which I
suspect explains why the cross-fit estimators show slightly larger empirical SEs than plain
AIPW/TMLE in our results. If you'd rather have textbook five-fold (nuisances trained on
four-fifths), I'd write that part myself rather than use his package — worth deciding when
we meet.

I'll have the Table 3 reproduction ready in the next day or so — happy to meet any time
after that.

Best wishes,
Nicolas
