# Email draft — supervisor status update

**STATUS: READY TO SEND.** Short version: code set up, paper read, running time is the
main point, open question about what he suggests + whether computers are available.
Deliberately does not report results and does not mention the n = 3,000 vs 500 point.

---

Subject: Zivich & Breskin reproduction — running time is a real problem

Hi [supervisor],

I've set up Zivich & Breskin's code from their GitHub and read their paper alongside it.

The running time is the thing I wanted to raise. Their simulation runs 6 estimators ×
3 specifications over 2,000 datasets of n = 3,000. The parametric specifications look
fine on my laptop, but the machine-learning ones are on a completely different scale:
the super learner is refit hundreds of times per dataset, so from some initial timing
those runs would take my laptop months of continuous computing — not days. Their own
README says these scripts "take a long time and should be broken into pieces to run in
parallel", so I think they ran them on a cluster.

What do you suggest I should do? Do you think I should try to use more powerful computers
that the University might have?

Best,
Nicolas
