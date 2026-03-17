# EXP4 Three-Way Comparison

- Source: `/home/hao/Desktop/Growpro/log/EXP4_riotee_data_ppfd_plus50_pn.csv`
- Input format: `sensor_ppfd` using `ppfd_adjusted`
- Nominal dt: `900 s`
- Replay segments: `3`
- PPFD lookup range: `0` to `600` umol m^-2 s^-1
- Clipped rows: `4` / `1081` (`0.370%`)

| scenario           | ppfd_umol_m2_s | solar_vol | energy_wh | segmented_umol_m2 | segmented_umol_m2_per_wh | mean_temp_c | max_temp_c | exogenous_umol_m2 | exogenous_umol_m2_per_wh |
| ------------------ | -------------- | --------- | --------- | ----------------- | ------------------------ | ----------- | ---------- | ----------------- | ------------------------ |
| dynamic_actual     | 182.656        |           | 8809.106  | 7630777.945       | 866.237                  | 31.132      | 37.225     | 7828590.336       | 888.693                  |
| constant_mean_ppfd | 182.656        | 1.231     | 7696.106  | 8834707.939       | 1147.945                 | 30.224      | 33.214     | 8926647.805       | 1159.891                 |
| constant_450_ppfd  | 450.000        | 1.614     | 18606.597 | 14046586.003      | 754.925                  | 35.213      | 38.339     | 15737675.198      | 845.812                  |
