# EXP4 09:00-01:00 PN-Energy-EP Table With Measured Energy

- Time window: `09:00:00` to next-day `01:00:00` (implemented as `time >= 09:00:00 or time < 01:00:00`).
- `dynamic_actual` and `constant_mean_ppfd` use summed `sensor.greenpi_c1_white_power` energy.
- `constant_450_ppfd` uses summed `sensor.greenpi_c1_redblue_power` energy.
- Energy source file: `/home/hao/Desktop/Growpro/log/EXP4_power_daily_average_new.csv`

| scenario           | pn_int_umol_m2 | pn_int_pct_vs_dynamic | energy_wh | energy_pct_vs_dynamic | ep_pn_per_wh | ep_pct_vs_dynamic | energy_source                   |
| ------------------ | -------------- | --------------------- | --------- | --------------------- | ------------ | ----------------- | ------------------------------- |
| dynamic_actual     | 7546930.373    | 0.000                 | 8255.431  | 0.000                 | 914.178      | 0.000             | sensor.greenpi_c1_white_power   |
| constant_mean_ppfd | 7858438.190    | 4.128                 | 8255.431  | 0.000                 | 951.911      | 4.128             | sensor.greenpi_c1_white_power   |
| constant_450_ppfd  | 10466471.964   | 38.685                | 13824.000 | 67.453                | 757.123      | -17.180           | sensor.greenpi_c1_redblue_power |
