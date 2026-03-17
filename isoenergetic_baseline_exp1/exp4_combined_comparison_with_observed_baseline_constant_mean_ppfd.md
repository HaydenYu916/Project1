# EXP4 Combined Comparison With Constant Mean PPFD As Baseline

- Percent columns use `constant_mean_ppfd = 0%`.

| scenario               | family             | pn_int_umol_m2 | pn_int_pct_vs_constant_mean_ppfd | energy_wh | energy_pct_vs_constant_mean_ppfd | ep_pn_per_wh | ep_pct_vs_constant_mean_ppfd | note                                       |
| ---------------------- | ------------------ | -------------- | -------------------------------- | --------- | -------------------------------- | ------------ | ---------------------------- | ------------------------------------------ |
| observed_original      | observed_exogenous | 7764179.743    | -3.417                           | 8255.431  | 0.000                            | 940.494      | -3.417                       | measured ppfd + observed temp              |
| observed_tempcopy_0914 | observed_exogenous | 7639974.308    | -4.962                           | 8255.431  | 0.000                            | 925.448      | -4.962                       | 11-09..11-14 temp copied from 11-02..11-07 |
| dynamic_actual         | full_replay        | 7543295.483    | -6.165                           | 8255.431  | 0.000                            | 913.737      | -6.165                       | 24h thermal replay, 09:00-01:00 evaluation |
| constant_mean_ppfd     | full_replay        | 8038897.998    | 0.000                            | 8255.431  | 0.000                            | 973.771      | 0.000                        | 24h thermal replay, 09:00-01:00 evaluation |
| constant_450_ppfd      | full_replay        | 9396147.551    | 16.884                           | 13824.000 | 67.453                           | 679.698      | -30.199                      | 24h thermal replay, 09:00-01:00 evaluation |
