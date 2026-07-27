# Email to Shaun — results of the reduced-design run

---

Subject: Reduced-design run finished — results

Dear Shaun,

I've finished running the design we agreed: the six methods, one split for the cross-fitting,
on a thousand simulated datasets. Every dataset ran with no failures, and I've stored all six
point estimates for each of the thousand datasets.

The results reproduce the paper's main message clearly. With the super learner doing the
nuisance estimation:

  Method                      Bias      Coverage(own SE)   Coverage(true SE)
  IPW                        +0.011          0.94               0.92
  G-computation              +0.026           --                0.64
  AIPW                       +0.004          0.91               0.95
  TMLE                       -0.001          0.90               0.95
  AIPW, 5-fold cross-fit     -0.003          0.94               0.95
  TMLE, 5-fold cross-fit     +0.001          0.94               0.96

(Truth = -0.108. "Own SE" uses each method's own standard error; "true SE" uses the empirical
standard deviation of the thousand estimates, as you suggested for g-computation.)

The headline points:

- G-computation is the most biased, and its intervals contain the truth only 64% of the time
  even using the true SE. Since that uses the true SE, the under-coverage is purely from bias,
  not from mis-estimating the variance — the clearest version of the plug-in problem.
- The doubly-robust methods (AIPW, TMLE) are essentially unbiased, and the cross-fit versions
  are the best behaved: smallest bias and coverage right at 95%.
- One extra thing that came out of computing both kinds of coverage: AIPW and TMLE without
  cross-fitting have own-SE coverage around 90%, a little low, and cross-fitting brings it back
  to 95%. So cross-fitting is visibly helping the variance estimation, not only the bias.

Everything (all point estimates, the summary, and the code) is on my GitHub fork. Happy to send
the tables or talk through any of it whenever suits you.

Best wishes,
Nicolas
