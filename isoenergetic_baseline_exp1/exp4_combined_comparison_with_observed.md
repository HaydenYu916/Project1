# EXP4 Combined Comparison With Observed Experiment

- `observed_*` rows are exogenous-temperature recomputations using measured PPFD.
- `dynamic_actual` / `constant_*` rows are full 24h thermal replay with 09:00-01:00 evaluation.

| scenario               | family             | pn_int_umol_m2 | pn_int_pct_vs_dynamic_actual | energy_wh | energy_pct_vs_dynamic_actual | ep_pn_per_wh | ep_pct_vs_dynamic_actual | note                                       |
| ---------------------- | ------------------ | -------------- | ---------------------------- | --------- | ---------------------------- | ------------ | ------------------------ | ------------------------------------------ |
| observed_original      | observed_exogenous | 7764179.743    | 2.928                        | 8255.431  | 0.000                        | 940.494      | 2.928                    | measured ppfd + observed temp              |
| observed_tempcopy_0914 | observed_exogenous | 7639974.308    | 1.282                        | 8255.431  | 0.000                        | 925.448      | 1.282                    | 11-09..11-14 temp copied from 11-02..11-07 |
| dynamic_actual         | full_replay        | 7543295.483    | 0.000                        | 8255.431  | 0.000                        | 913.737      | 0.000                    | 24h thermal replay, 09:00-01:00 evaluation |
| constant_mean_ppfd     | full_replay        | 8038897.998    | 6.570                        | 8255.431  | 0.000                        | 973.771      | 6.570                    | 24h thermal replay, 09:00-01:00 evaluation |
| constant_450_ppfd      | full_replay        | 9396147.551    | 24.563                       | 13824.000 | 67.453                       | 679.698      | -25.613                  | 24h thermal replay, 09:00-01:00 evaluation |
