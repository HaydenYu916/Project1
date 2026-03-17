# EXP4 Eval 09:00-01:00 Full-Replay PN-Energy-EP Table

- Thermal replay uses full 24h sequence.
- Metrics and baseline construction use only `09:00:00` to next-day `01:00:00`.
- Night rows remain in replay with `solar_vol=0`, so the thermal model switches to cooling.

| scenario           | pn_int_umol_m2 | pn_int_pct_vs_dynamic | energy_wh | energy_pct_vs_dynamic | ep_pn_per_wh | ep_pct_vs_dynamic | energy_source                   |
| ------------------ | -------------- | --------------------- | --------- | --------------------- | ------------ | ----------------- | ------------------------------- |
| dynamic_actual     | 7543295.483    | 0.000                 | 8255.431  | 0.000                 | 913.737      | 0.000             | sensor.greenpi_c1_white_power   |
| constant_mean_ppfd | 8038897.998    | 6.570                 | 8255.431  | 0.000                 | 973.771      | 6.570             | sensor.greenpi_c1_white_power   |
| constant_450_ppfd  | 9396147.551    | 24.563                | 13824.000 | 67.453                | 679.698      | -25.613           | sensor.greenpi_c1_redblue_power |
