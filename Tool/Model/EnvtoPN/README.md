## EnvtoPN

This folder is a portable Pn prediction package.

Contents:
- `predict_env_to_pn.py`: prediction script
- `model/best_model_package.joblib`: trained model package
- `model/best_model_metadata.json`: model metadata
- `input/new_samples.csv`: example input CSV

Default CSV mode:

```bash
python predict_env_to_pn.py
```

Custom CSV input and output:

```bash
python predict_env_to_pn.py --input-csv input/new_samples.csv --output-csv predictions.csv
```

Single-value mode:

```bash
python predict_env_to_pn.py --T 18 --CO2 400 --RB 0.75 --PPFD 300
```

Required input fields:
- `T`
- `CO2`
- `R:B`
- `PPFD`

The script appends one prediction column:
- `Pn_pred`

In single-value mode, the script prints the prediction result directly to the terminal.
