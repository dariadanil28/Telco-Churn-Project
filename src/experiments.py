from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.model import build_model
from src.plots import plot_confusion_pair


def build_historical_model(X_train):
    template = build_model(X_train)

    return Pipeline([
        (
            "preprocessor",
            clone(template.named_steps["preprocessor"])
        ),
        ("classifier", LogisticRegression(max_iter=1000))
    ])


def save_experiment_results(
    model,
    X_test,
    y_test,
    title,
    filename,
    output_dir
):
    predictions = model.predict(X_test)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=["No", "Yes"]
    )

    plot_confusion_pair(
        matrix,
        title,
        output_dir / f"{filename}.png"
    )

    report = classification_report(
        y_test,
        predictions,
        labels=["No", "Yes"],
        digits=3,
        zero_division=0
    )

    (output_dir / f"{filename}_report.txt").write_text(
        report,
        encoding="utf-8"
    )


def run_historical_experiments(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    historical_data = df.copy()

    historical_data["TotalCharges"] = pd.to_numeric(
        historical_data["TotalCharges"],
        errors="coerce"
    ).fillna(0)

    X = historical_data.drop(columns=["Churn", "customerID"])
    y = historical_data["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Original logistic regression with TotalCharges and default threshold.
    baseline_model = build_historical_model(X_train)
    baseline_model.fit(X_train, y_train)

    save_experiment_results(
        baseline_model,
        X_test,
        y_test,
        "Historical Baseline Logistic Regression — Threshold 0.50",
        "baseline_logistic_confusion_matrix",
        output_dir
    )

    # Earlier experiment using duplicated churners with switched gender.
    X_train_new = X_train.drop(columns=["TotalCharges"]).copy()
    X_test_new = X_test.drop(columns=["TotalCharges"]).copy()

    training_data = X_train_new.copy()
    training_data["Churn"] = y_train

    yes_customers = training_data[
        training_data["Churn"] == "Yes"
    ]
    no_customers = training_data[
        training_data["Churn"] == "No"
    ]

    additional_customers = yes_customers.sample(
        n=len(no_customers) - len(yes_customers),
        replace=True,
        random_state=42
    ).copy()

    additional_customers["gender"] = (
        additional_customers["gender"].map({
            "Male": "Female",
            "Female": "Male"
        })
    )

    balanced_training = pd.concat(
        [training_data, additional_customers],
        ignore_index=True
    ).sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    X_balanced = balanced_training.drop(columns=["Churn"])
    y_balanced = balanced_training["Churn"]

    historical_balanced_model = build_historical_model(X_balanced)
    historical_balanced_model.fit(X_balanced, y_balanced)

    save_experiment_results(
        historical_balanced_model,
        X_test_new,
        y_test,
        "Historical Gender-Switch Experiment — Threshold 0.50",
        "gender_switch_confusion_matrix",
        output_dir
    )