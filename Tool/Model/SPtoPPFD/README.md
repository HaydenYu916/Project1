## SPtoPPFD

This folder is a portable PPFD prediction package.

Contents:
- `predict_sp_to_ppfd.py`: prediction script
- `model/best_model_package.joblib`: trained model package
- `model/best_model_metadata.json`: model metadata
- `input/new_samples.csv`: example input CSV

Default CSV mode:

```bash
python predict_sp_to_ppfd.py
```

CSV mode:

```bash
python predict_sp_to_ppfd.py --input-csv input/new_samples.csv --output-csv predictions.csv
```

Single-value mode:

```bash
python predict_sp_to_ppfd.py \
  --sp-415 84 \
  --sp-445 769 \
  --sp-480 450 \
  --sp-515 63.5 \
  --sp-555 2289.5 \
  --sp-590 121 \
  --sp-630 84 \
  --sp-680 86
```

The input CSV must contain these columns:
- `sp_415_mean`
- `sp_445_mean`
- `sp_480_mean`
- `sp_515_mean`
- `sp_555_mean`
- `sp_590_mean`
- `sp_630_mean`
- `sp_680_mean`

The script keeps all original columns and appends one new column:
- `PPFD_pred`

In single-value mode, the script prints the prediction result directly to the terminal.
