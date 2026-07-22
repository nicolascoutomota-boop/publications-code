# Reply to Shaun's proposal — READY TO SEND

Measured 2026-07-21 on idle machine: one super-learner fit (K=10, n=3000) = 13 s.
Fits/dataset = 6 + 20k (k = cross-fit repetitions). Original design = 1,456.
  k=1: 26 fits = 1.8% | 1000 sets ~1 day, 2000 ~2 days (4 cores)
  k=3: 66 fits = 4.5% | 1000 ~2.5 days, 2000 ~5 days
  k=5: 106 fits = 7.3% | 1000 ~4 days, 2000 ~8 days

---

Dear Shaun,

Sorry for the late reply. This all makes sense, and it takes away almost all of the computing
problem.

**On the seed:** yes, you're right. Their data is fixed, but the analysis is not — the random
forest and the neural net both make random choices as they run, and the authors never fixed
those. So their machine-learning results can't be repeated exactly, by them or by anyone.
I'll set a seed in my own code, so at least ours can be repeated later.

**One idea:** since the data gets split into folds at random, do you think it would be worth
doing that split a few times over, each time slightly differently, and combining the results
— so the answer doesn't rest on a single random split? Or would you rather keep it to once?
I wasn't sure which you'd prefer.

**On timing:** from timing a single fit and scaling up, if we do the split once a thousand
datasets would take my laptop about a day, and two thousand about two days. If we do it five
times, nearer four days and a week. Either way, far below their original, which would have
run for months.

Best wishes,
Nicolas
