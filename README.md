# Telco Customer Churn

A Python project for identifying customers likely to churn and
exploring customer characteristics associated with churn.

## Approach

- Stratified 80/20 training and test split.
- Numeric imputation and standardization.
- Categorical imputation and one-hot encoding.
- Random oversampling within the training pipeline.
- Logistic regression.
- A fixed decision threshold of 0.35.
- Five-fold stratified cross-validation on the training set.

The final model excludes customerID and TotalCharges from its features.
Churn is the prediction target.

## Project files

- `main.py` runs the workflow.
- `src/data.py` loads and splits the dataset.
- `src/model.py` defines the model pipeline.
- `src/evaluate.py` evaluates predictions and extracts coefficients.
- `src/plots.py` saves charts.
- `notebooks/` preserves exploratory work.
- `outputs/` contains generated results.

## Run on macOS

Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in `data/`.

From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

The dataset is the Telco Customer Churn dataset used in the accompanying
exploratory notebook. The CSV is excluded from version control.

## Results

Earlier notebook runs produced approximately:

| Metric | Cross-validation mean | Test set |
|---|---:|---:|
| Recall for churn | 89.4% | 89.8% |
| Precision for churn | 44.8% | 44.9% |
| Accuracy | 67.9% | 68.1% |

Run the project to generate results for your installed package versions.

## Interpretation and limitations

The 0.35 threshold prioritizes catching churners while increasing false
alarms. It was informed by earlier experiments on this dataset.

Cross-validation reuses training data involved in model and threshold
selection. It is a consistency check, rather than an independent
evaluation of the entire selection process.

The test set was inspected during exploratory development, so its
results should be interpreted with that history in mind.

Oversampling changes the training class proportions. Model scores
have not been calibrated as real-world churn probabilities.

Coefficient plots describe associations within the fitted model.
They do not establish causes of churn, and coefficient magnitudes
depend on feature encoding, scaling, and correlated features.

The exploratory notebook records earlier experiments, including
alternative models, feature engineering, and PCA.