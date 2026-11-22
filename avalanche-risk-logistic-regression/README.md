# Avalanche Occurrence Prediction – Logistic Regression

A machine learning project that uses **Logistic Regression** to predict whether an **avalanche occurs** on a slope from **Slope Angle** and **Fresh Snowfall (24 h)**. Covers EDA, scaling, training, evaluation, cross-validation, decision-boundary plotting, prediction and model saving.


## Dataset
**Avalanche_Events.csv** – 600 rows, 6 columns.

| Column | Description |
|---|---|
| Event_ID | Row identifier (not a feature) |
| Region | Alps / Rockies / Himalaya / Andes (no effect by design) |
| Slope_Angle | Slope steepness in degrees |
| New_Snowfall_cm | Fresh snowfall in the last 24 h (cm) |
| Temperature_C | Air temperature (°C) – unused by the 2-D model |
| Avalanche | Target: `0` = no avalanche, `1` = avalanche (181 positive / 419 negative) |

## Results (test set, 120 samples)
| Metric | Value |
|---|---|
| Accuracy | 0.8167 |
| Precision | 0.7917 |
| Recall | 0.5278 |
| F1 Score | 0.6333 |
| ROC AUC | 0.8409 |
| 5-fold CV accuracy | 0.8067 |

Full outputs and plots are stored inside the notebook.

## Run it
```
pip install -r requirements.txt
python generate_dataset.py          # optional: regenerates the CSV
jupyter notebook Avalanche_Logistic_Regression.ipynb
```

## Structure
```
avalanche-risk-logistic-regression/
├── Avalanche_Events.csv
├── generate_dataset.py
├── Avalanche_Logistic_Regression.ipynb
├── logistic_regression_model.pkl
├── standard_scaler.pkl
├── requirements.txt
└── README.md
```
