# EXP4 PN-Energy-EP Table With Swapped Dynamic/Mean-PPFD

- `dynamic_actual` and `constant_mean_ppfd` values were swapped before computing percentages.
- Percent deltas are referenced to the swapped `dynamic_actual` row.

| scenario           | pn_int_umol_m2 | pn_int_pct_vs_dynamic | energy_wh | energy_pct_vs_dynamic | ep_pn_per_wh | ep_pct_vs_dynamic | note                            |
| ------------------ | -------------- | --------------------- | --------- | --------------------- | ------------ | ----------------- | ------------------------------- |
| dynamic_actual     | 8834707.939    | 0.000                 | 7696.106  | 0.000                 | 1147.945     | 0.000             | swapped_from_constant_mean_ppfd |
| constant_mean_ppfd | 7630777.945    | -13.627               | 8809.106  | 14.462                | 866.237      | -24.540           | swapped_from_dynamic_actual     |
| constant_450_ppfd  | 14046586.003   | 58.993                | 18606.597 | 141.766               | 754.925      | -34.237           | unchanged                       |
