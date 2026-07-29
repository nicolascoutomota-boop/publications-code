# Reply to Shaun's histogram request (attach histograms_ate.pdf)

---

Subject: Re: histograms

Dear Shaun,

Attached are the histograms for the completed six-method study: the 1,000 ATE estimates
for all six methods and all three specifications, each panel with a fitted normal curve
and a dashed line at the truth.

For this study the ATE is the only estimand I can plot, since it stored only the
difference and not the two means, as we discussed. The E(Y^1) and E(Y^0) histograms will
come from the g-computation studies, which store all three estimands for every dataset.
So each of those studies will arrive with its Table 3s and histograms for E(Y^1), E(Y^0)
and the ATE. Study 3 should be done today.

On what the attached ones show: the machine-learning panels all look close to normal,
with no outliers beyond four standard deviations. A few parametric panels deviate a
little: IPW under main-effects has a longer left tail with three estimates beyond four
SDs, and single cross-fit TMLE under the true specification is somewhat right-skewed.
G-computation's bias under machine learning is directly visible: its histogram sits to
the right of the truth.

Best wishes,
Nicolas
