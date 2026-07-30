# Delivery email: all three g-computation studies
# Attach: study{1,2,3}_histograms.pdf + study{1,2,3}_tables.csv

---

Subject: All three studies finished

Dear Shaun,

All three simulation studies are finished, each on the same 1,000 datasets with no
failures. For each study I attach its histograms as a PDF and its tables as a CSV,
covering E(Y^1), E(Y^0) and the ATE, with confidence intervals from the empirical SE
throughout.

A few headline numbers for g-computation under the data-adaptive specification, against
the baseline from the earlier study (ATE bias +0.026, coverage 64% with the full super
learner at n=3000):

Study 1 (random forest alone): ATE bias +0.039, coverage 30%. The random forest by itself
is clearly worse than the super learner.

Study 2 (n=1500): ATE bias +0.037, coverage 60%. Halving the sample size makes the bias
worse.

Study 3 (two separate super learners): ATE bias +0.017, coverage 85%. This helps. Looking
at the two means separately, almost all of the bias is in E(Y^0) (bias -0.017, coverage
69%), while E(Y^1) is essentially unbiased with 96% coverage.

In studies 1 and 2 the two means are biased in opposite directions (E(Y^1) upward, E(Y^0)
downward), so their biases add up in the ATE.

Happy to go through all of it on Friday.

Best wishes,
Nicolas
