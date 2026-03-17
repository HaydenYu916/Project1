# EXP4 Comparison In Mol Units

- `Pn int` is reported as `mol m^-2`.
- `EP` is reported as `mmol Wh^-1`.
- Percent columns use `constant_mean_ppfd = 0%`.

| scenario               | family             | pn_int_mol_m2 | pn_int_pct_vs_constant_mean_ppfd | energy_wh | energy_pct_vs_constant_mean_ppfd | ep_mmol_per_wh | ep_pct_vs_constant_mean_ppfd | note                                       |
| ---------------------- | ------------------ | ------------- | -------------------------------- | --------- | -------------------------------- | -------------- | ---------------------------- | ------------------------------------------ |
| observed_original      | observed_exogenous | 7.764         | -3.417                           | 8255.431  | 0.000                            | 0.940          | -3.417                       | measured ppfd + observed temp              |
| observed_tempcopy_0914 | observed_exogenous | 7.640         | -4.962                           | 8255.431  | 0.000                            | 0.925          | -4.962                       | 11-09..11-14 temp copied from 11-02..11-07 |
| dynamic_actual         | full_replay        | 7.543         | -6.165                           | 8255.431  | 0.000                            | 0.914          | -6.165                       | 24h thermal replay, 09:00-01:00 evaluation |
| constant_mean_ppfd     | full_replay        | 8.039         | 0.000                            | 8255.431  | 0.000                            | 0.974          | 0.000                        | 24h thermal replay, 09:00-01:00 evaluation |
| constant_450_ppfd      | full_replay        | 9.396         | 16.884                           | 13824.000 | 67.453                           | 0.680          | -30.199                      | 24h thermal replay, 09:00-01:00 evaluation |
