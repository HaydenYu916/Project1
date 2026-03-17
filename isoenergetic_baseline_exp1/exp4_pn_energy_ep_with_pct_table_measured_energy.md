# EXP4 PN-Energy-EP Table With Measured Energy

- `dynamic_actual` and `constant_mean_ppfd` use summed `sensor.greenpi_c1_white_power` energy.
- `constant_450_ppfd` uses summed `sensor.greenpi_c1_redblue_power` energy.
- Energy source file: `/home/hao/Desktop/Growpro/log/EXP4_power_daily_average_new.csv`

| scenario           | pn_int_umol_m2 | pn_int_pct_vs_dynamic | energy_wh | energy_pct_vs_dynamic | ep_pn_per_wh | ep_pct_vs_dynamic | energy_source                   |
| ------------------ | -------------- | --------------------- | --------- | --------------------- | ------------ | ----------------- | ------------------------------- |
| dynamic_actual     | 7630777.945    | 0.000                 | 8255.431  | 0.000                 | 924.334      | 0.000             | sensor.greenpi_c1_white_power   |
| constant_mean_ppfd | 8834707.939    | 15.777                | 8255.431  | 0.000                 | 1070.169     | 15.777            | sensor.greenpi_c1_white_power   |
| constant_450_ppfd  | 14046586.003   | 84.078                | 13824.000 | 67.453                | 1016.101     | 9.928             | sensor.greenpi_c1_redblue_power |
