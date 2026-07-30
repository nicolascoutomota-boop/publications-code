# Delivery email: all three g-computation studies
# Attach: study{1,2,3}_histograms.pdf + study{1,2,3}_tables.csv

---

Subject: All three studies finished

Dear Shaun,

All three studies are finished, each on the same 1,000 datasets with no failures. Tables
and histograms for E(Y^1), E(Y^0) and the ATE are attached for each.

In short, for g-computation under the data-adaptive specification:

Study 1 (random forest alone): ATE bias +0.039, coverage 30%.
Study 2 (n=1500): ATE bias +0.037, coverage 60%.
Study 3 (two separate super learners): ATE bias +0.017, coverage 85%. Almost all of its
bias is in E(Y^0); E(Y^1) is essentially unbiased.

Is there anything I should start on next, for example the further study you mentioned you
would work out with some algebra? Otherwise I'll begin drafting the presentation along the
structure you suggested, and we can go through everything on Friday.

Best wishes,
Nicolas
