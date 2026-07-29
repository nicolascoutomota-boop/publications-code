# Short email to Shaun with the reproduced Table 3

---

Subject: Reproduced Table 3

Dear Shaun,

Friday 4pm works. Thank you for the presentation details, that helps a lot.

Here is the reproduced Table 3 (1,000 datasets, truth = -0.108). The two changes are as
agreed: g-computation's confidence intervals use the empirical SE (so it has no ASE
column entry), and the cross-fit rows are single cross-fit (SC) rather than double.

                          Bias    RMSE     ASE     ESE     CLD  Coverage
  G-computation
    True                +0.000   0.017      --   0.017   0.068    94.8%
    Main-effects        -0.022   0.028      --   0.018   0.069    76.8%
    Machine learning    +0.026   0.031      --   0.017   0.065    64.0%
  IPW
    True                +0.007   0.025   0.025   0.024   0.097    94.4%
    Main-effects        -0.022   0.032   0.023   0.023   0.091    86.7%
    Machine learning    +0.011   0.024   0.023   0.021   0.089    93.7%
  AIPW
    True                +0.000   0.021   0.020   0.021   0.078    94.1%
    Main-effects        -0.016   0.026   0.020   0.020   0.077    85.4%
    Machine learning    +0.004   0.020   0.017   0.019   0.065    90.8%
  TMLE
    True                +0.000   0.021   0.020   0.021   0.077    93.8%
    Main-effects        -0.017   0.025   0.019   0.018   0.076    85.9%
    Machine learning    -0.001   0.020   0.016   0.020   0.065    90.1%
  SC-AIPW
    True                +0.000   0.025   0.024   0.025   0.094    95.1%
    Main-effects        -0.011   0.037   0.032   0.035   0.125    90.4%
    Machine learning    -0.003   0.024   0.022   0.023   0.087    93.5%
  SC-TMLE
    True                +0.003   0.025   0.023   0.024   0.090    94.6%
    Main-effects        -0.017   0.026   0.028   0.020   0.109    92.1%
    Machine learning    +0.001   0.020   0.020   0.020   0.080    93.8%

On the coverage you are most interested in: g-computation under machine learning covers
64%, and since its intervals are built from the empirical SE, that under-coverage is
entirely due to bias, not to a bad variance estimate. The parametric rows are very close
to the paper's published values, and g-computation's machine learning bias of 0.026
matches theirs exactly.

Study 3 is running and should be finished today; studies 1 and 2 will follow.

Best wishes,
Nicolas
