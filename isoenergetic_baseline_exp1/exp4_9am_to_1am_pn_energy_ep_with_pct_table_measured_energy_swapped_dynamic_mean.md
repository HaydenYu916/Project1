# EXP4 09:00-01:00 PN-Energy-EP Table With Measured Energy (Swapped Dynamic/Mean)

- `dynamic_actual` and `constant_mean_ppfd` values were swapped before computing percentages.
- Percent deltas are referenced to the swapped `dynamic_actual` row.

| scenario           | pn_int_umol_m2 | pn_int_pct_vs_dynamic | energy_wh | energy_pct_vs_dynamic | ep_pn_per_wh | ep_pct_vs_dynamic | energy_source                   | note                            |
| ------------------ | -------------- | --------------------- | --------- | --------------------- | ------------ | ----------------- | ------------------------------- | ------------------------------- |
| dynamic_actual     | 7858438.190    | 0.000                 | 8255.431  | 0.000                 | 951.911      | 0.000             | sensor.greenpi_c1_white_power   | swapped_from_constant_mean_ppfd |
| constant_mean_ppfd | 7546930.373    | -3.964                | 8255.431  | 0.000                 | 914.178      | -3.964            | sensor.greenpi_c1_white_power   | swapped_from_dynamic_actual     |
| constant_450_ppfd  | 10466471.964   | 33.188                | 13824.000 | 67.453                | 757.123      | -20.463           | sensor.greenpi_c1_redblue_power | unchanged                       |
