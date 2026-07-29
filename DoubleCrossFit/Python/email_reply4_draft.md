# Combined reply to Shaun — final clean version, ready to send

---

Subject: Re: next simulation studies

Dear Shaun,

Answering both your emails here, question by question.

**"Are you getting invited to the career talks?"**
Yes, and I have already been to some of them.

**"Choose a time to meet."**
Could we meet on Friday? I am available any time after 4 pm UK time. Before then, could you
also tell me a bit about the presentations on 11 August: what I should present, how long it
should be, what format, and who will be in the audience?

**"How long were you expecting the study you just did to take?"**
About a day or two, the same as you. It took four because my laptop slows down a lot when
it computes non-stop for days, and some time was also lost when runs got interrupted. I
have fixed the interruption problem, so that should not happen again.

**"Is it correct that the random forest study would take almost as long?"**
No, that was a misunderstanding. The random forest on its own is about twenty times faster
than the super learner, so that study is quick.

**"How long will the three new studies take?"**
Your estimate is right. G-computation alone comes to about 4% of the last study, so if
anything the studies will be a little quicker than you expected. My estimates: study 3
(two super learners) about half a day, study 1 (random forest) under an hour, study 2
(n=1500) two to three hours. All three together should be about a day of computing.

**"Reproduce Z&B's Table 3 with the two changes."**
Done, for the ATE. The parametric rows match the paper's numbers almost exactly, and
g-computation's machine learning bias of 0.026 matches theirs exactly. I will bring the
table to the meeting.

**"Can your stored results also produce the E(Y^1) and E(Y^0) versions? If not, let me know."**
Those two they cannot produce. For every method I stored the difference, which is why the
ATE table above was possible, but not the two means separately. For g-computation this is
easy to fix, so the three new studies will save E(Y^1), E(Y^0) and the ATE from the start.
Getting the two means for the other five methods would need the long multi-day re-run, so I
will assume you do not want that.

**"Start with study 3."**
Starting it now. Two small notes. First, I have to write the two-model code myself, because
their code fits a single outcome model. Second, for the n=1500 study I plan to use the
first 1,500 rows of each existing dataset, since the rows are independent draws; tell me if
you would rather I generated fresh data.

Best wishes,
Nicolas
