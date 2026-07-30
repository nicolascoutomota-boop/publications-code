# Reply: original-design arms done + feasibility of studies 3c/4/5
# Attach: baseline_tables.csv + baseline_histograms.pdf

---

Subject: Re: further studies

Dear Shaun,

Yes. I ran the original-design g-computation with both means recorded, on the same 1,000
datasets; its tables and histograms are attached. Under machine learning its bias splits
across the two means (E(Y^1) +0.014, E(Y^0) -0.013), which add up in the ATE.

The three new studies are feasible. Your count of 12 fits is right if I fit each nuisance
model once per dataset and share it across the estimators: one propensity model and one
untreated-only outcome model on the full sample, plus one of each per fold for the
cross-fit versions. I would write the E(Y^0) estimators myself on top of those shared
fits, since the packages only compute the difference. No individual with X=1 will be used
to estimate E(Y|X=0,Z) anywhere.

On timing: study 3 continued should be about a day of computing, study 4 a bit less, and
study 5 an hour or two. Running them in your order, everything should be done over the
weekend.

See you tomorrow at 4pm.

Best wishes,
Nicolas
