# Combined reply (main mail + p.s.) — attach baseline_tables.csv + baseline_histograms.pdf

---

Subject: Re: further studies

Dear Shaun,

Yes. I ran the original-design g-computation with both means recorded, on the same 1,000
datasets; its tables and histograms are attached. Under machine learning its bias splits
across the two means (E(Y^1) +0.014, E(Y^0) -0.013), which add up in the ATE.

The three new studies are feasible, and study 3 continued is already running. Your count of
12 fits is right: I fit each nuisance model once per dataset and share it across the
estimators, one propensity model and one untreated-only outcome model on the full sample,
plus one of each per fold for the cross-fit versions. No individual with X=1 is used
anywhere to estimate E(Y|X=0,Z), and E(Y|X=1,Z) is never estimated at all. I wrote the
E(Y^0) estimators myself, since the packages only compute the difference; I checked them
against Zivich and Breskin's own code on identical inputs and they agree to machine
precision.

On your p.s.: already done that way. The program runs all six estimators together, and
g-computation comes almost free, as its estimate is just the average of the same
E(Y|X=0,Z) fit that AIPW and TMLE use.

On timing: from the running study's actual pace, study 3 continued needs about a day and a
half, study 4 about a day, and study 5 an hour or two. Everything should be finished over
the weekend.

See you tomorrow at 4pm.

Best wishes,
Nicolas
