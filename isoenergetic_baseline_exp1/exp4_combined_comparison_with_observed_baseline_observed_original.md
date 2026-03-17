# EXP4 Combined Comparison With Observed Original As Baseline

- Percent columns use `observed_original = 0%`.

| scenario               | family             | pn_int_umol_m2 | pn_int_pct_vs_observed_original | energy_wh | energy_pct_vs_observed_original | ep_pn_per_wh | ep_pct_vs_observed_original | note                                       |
| ---------------------- | ------------------ | -------------- | ------------------------------- | --------- | ------------------------------- | ------------ | --------------------------- | ------------------------------------------ |
| observed_original      | observed_exogenous | 7764179.743    | 0.000                           | 8255.431  | 0.000                           | 940.494      | 0.000                       | measured ppfd + observed temp              |
| observed_tempcopy_0914 | observed_exogenous | 7639974.308    | -1.600                          | 8255.431  | 0.000                           | 925.448      | -1.600                      | 11-09..11-14 temp copied from 11-02..11-07 |
| dynamic_actual         | full_replay        | 7543295.483    | -2.845                          | 8255.431  | 0.000                           | 913.737      | -2.845                      | 24h thermal replay, 09:00-01:00 evaluation |
| constant_mean_ppfd     | full_replay        | 8038897.998    | 3.538                           | 8255.431  | 0.000                           | 973.771      | 3.538                       | 24h thermal replay, 09:00-01:00 evaluation |
| constant_450_ppfd      | full_replay        | 9396147.551    | 21.019                          | 13824.000 | 67.453                          | 679.698      | -27.730                     | 24h thermal replay, 09:00-01:00 evaluation |
