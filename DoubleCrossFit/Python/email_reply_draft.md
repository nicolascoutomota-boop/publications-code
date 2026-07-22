# Reply to Shaun — READY TO SEND

Plain-text formatting (no markdown tables) so it survives any email client.
Every figure verified in source or measured on this machine; provenance in `notes.md`.

---

Subject: Re: Zivich & Breskin reproduction — running time is a real problem

Dear Shaun,

Sorry for the unclarities in my last email — let me answer all four questions properly.

**1. Six estimators, because double cross-fitting comes in two versions.**

Their code has six separate estimator classes, and six corresponding simulation scripts:
g-computation, IPW, AIPW, TMLE, double cross-fit AIPW, and double cross-fit TMLE. Your
five becomes six once "double-cross-fit" is split into its AIPW and TMLE versions. Six
estimators times three specifications is where the 18 runs come from.

**2. On whether the huge computing time is mostly due to double cross-fitting — yes,
mostly.**

Super-learner fits per dataset: IPW 1, AIPW 2, TMLE 2, g-computation 251, DC-AIPW 600,
DC-TMLE 600. The two double cross-fit ones are 1,200 of the 1,456. Their fits each use only
a third of the data, since the method splits the sample three ways and fits each model on
one part, so in time they come to about three-quarters of the total and g-computation to
about a fifth.

G-computation is the exception because it has to be bootstrapped, and their script uses 250
replicates per dataset. That alone would take my laptop about eight or nine months, so
dropping double cross-fitting would not be enough on its own.

I got these numbers from their own single-dataset example script: just under eight hours for
one dataset, times 2,000.

**3. The three specifications are their "setup" variable.**

Each script has a line "setup = 1" near the top which selects how the nuisance models are
estimated:

  setup = 1   correctly specified parametric models — they match the data-generating
              equations term for term, including the thresholds, the squared age term
              and the treatment-by-LDL interaction
  setup = 2   main-terms-only parametric models, i.e. deliberately misspecified
  setup = 3   the super learner

You run each script three times, changing that one digit. Only setup 3 is expensive.

**4. On which learners are slow.**

I had an AI agent time them on one dataset. The random forest (500 trees) dominates, at
roughly 80-85% of the super-learner's runtime, and that holds at every sample size I tried,
from 250 up to 12,000. Everything else is minor: the two GAMs come to under 10%, logistic
regression about 2%, and the neural net is small but erratic — anywhere between under 1% and
10%, depending on how quickly its optimiser converges. So the random forest is the real
cost, and removing the neural net would save very little.

One thing worth flagging for later: the random forest builds its 500 trees whatever the
sample size, so it is only about twice as fast at n = 250 as at n = 3,000, not twelve times.
Smaller sample sizes will help less than one might expect.

**On the computing.**

Reproducing everything exactly would be around 30,000 core-hours, but a few changes would
bring it down sharply — for instance fewer sample-splits in the double cross-fit runs (they
use 100), 1,000 datasets instead of 2,000, or a smaller random forest. Together these could
bring it down to roughly 2,000 core-hours — days rather than months, though probably a week
or so on an ordinary desktop rather than an afternoon. That figure is an AI agent's estimate,
combining the reductions arithmetically rather than measuring them, so I'd want to time one
dataset under whatever settings you choose before promising a schedule.

Each of these means running something different from what the authors published, so it would
no longer be an exact reproduction — which is why I'd rather leave the choice to you. Tell me
which you're comfortable with and I'll run that grid and document what changed.

Meanwhile the parametric specifications are cheap and I'm working through them on my laptop
now — those need no decisions from you.

Do let me know if I've missed any of your questions, or if anything needs more detail.

Best wishes,
Nicolas
