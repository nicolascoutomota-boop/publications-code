# Reply to Shaun's histogram request (attach histograms_ate.pdf)

Verified: all 18 panels match Table 3 exactly; skew/kurtosis/outliers computed per panel
(scipy). ML panels: skew<=0.13, zero |z|>4, Shapiro p>0.2. Deviations: IPW/main (skew
-0.44, three |z|>4), SC-TMLE/true (skew +0.58), SC-AIPW/main (kurtosis +0.92).

---

Subject: Re: histograms

Dear Shaun,

Attached are the histograms of the 1,000 ATE estimates for all six methods and all three
specifications. Each panel has a fitted normal curve and a dashed line at the truth.

The machine-learning panels all look close to normal, with no outliers beyond four
standard deviations. A few parametric panels deviate a little: IPW under main-effects has
a longer left tail with three estimates beyond four SDs, and single cross-fit TMLE under
the true specification is somewhat right-skewed. G-computation's bias under machine
learning is directly visible: its histogram sits to the right of the truth.

The E(Y^1) and E(Y^0) histograms for g-computation will follow with study 3's results.

Best wishes,
Nicolas
