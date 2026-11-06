import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import StratifiedKFold


def predict_churn(model, X, threshold):
    yes_column = list(model.classes_).index("Yes")
    scores = model.predict_proba(X)[:, yes_column]
    predictions = np.where(scores > threshold, "Yes", "No")

    return scores, predictions


def calculate_metrics(y_true, predictions):
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision (Yes)": precision_score(
            y_true, predictions, pos_label="Yes", zero_division=0
        ),
        "Recall (Yes)": recall_score(
            y_true, predictions, pos_label="Yes", zero_division=0
        ),
        "F1 (Yes)": f1_score(
            y_true, predictions, pos_label="Yes", zero_division=0
        )
    }


def cross_validate_model(model, X, y, threshold):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    fold_results = []

    for fold, (train_index, validation_index) in enumerate(
        cv.split(X, y), start=1
    ):
        fold_model = clone(model)

        fold_model.fit(
            X.iloc[train_index],
            y.iloc[train_index]
        )

        _, predictions = predict_churn(
            fold_model,
            X.iloc[validation_index],
            threshold
        )

        metrics = calculate_metrics(
            y.iloc[validation_index],
            predictions
        )

        fold_results.append({"Fold": fold, **metrics})

    return pd.DataFrame(fold_results).set_index("Fold")


def evaluate_test_set(model, X_test, y_test, threshold):
    scores, predictions = predict_churn(
        model, X_test, threshold
    )

    metrics = calculate_metrics(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        labels=["No", "Yes"],
        digits=3,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=["No", "Yes"]
    )

    return metrics, report, matrix, scores, predictions


def get_coefficients(model):
    feature_names = model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    classifier = model.named_steps["classifier"]

    if list(classifier.classes_) != ["No", "Yes"]:
        raise ValueError("Expected classifier classes ['No', 'Yes'].")

    coefficients = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": classifier.coef_[0]
    })

    return coefficients.sort_values(
        "Coefficient",
        ascending=False
    )