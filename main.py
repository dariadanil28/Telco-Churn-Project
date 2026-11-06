import json
from pathlib import Path
from src.experiments import run_historical_experiments
from src.plots import generate_exploratory_plots
import pandas as pd

from src.data import load_data, split_data
from src.evaluate import (
    cross_validate_model,
    evaluate_test_set,
    get_coefficients
)
from src.model import build_model
from src.plots import (
    plot_churn_distribution,
    plot_coefficients,
    plot_confusion_matrix
)


THRESHOLD = 0.35


def main():
    project_dir = Path(__file__).resolve().parent
    csv_path = (
        project_dir
        / "data"
        / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    output_dir = project_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    X_train, X_test, y_train, y_test = split_data(df)

    print(f"Dataset loaded: {df.shape[0]} customers")
    print(f"Training customers: {len(X_train)}")
    print(f"Test customers: {len(X_test)}")
    print(f"Decision threshold: {THRESHOLD}")

    model = build_model(X_train)

    print("\nRunning five-fold cross-validation...")

    cv_results = cross_validate_model(
        model,
        X_train,
        y_train,
        THRESHOLD
    )

    cv_summary = cv_results.agg(["mean", "std"])

    print("\nResults per fold:")
    print(cv_results.round(3).to_string())

    print("\nCross-validation summary:")
    print(cv_summary.round(3).to_string())

    cv_results.to_csv(output_dir / "cross_validation_folds.csv")
    cv_summary.to_csv(output_dir / "cross_validation_summary.csv")

    print("\nTraining final model...")
    model.fit(X_train, y_train)

    metrics, report, matrix, scores, predictions = evaluate_test_set(
        model,
        X_test,
        y_test,
        THRESHOLD
    )

    print("\nTest-set classification report:")
    print(report)

    (output_dir / "classification_report.txt").write_text(
        report,
        encoding="utf-8"
    )

    evaluation_details = {
        "threshold": THRESHOLD,
        "training_customers": len(X_train),
        "test_customers": len(X_test),
        "metrics": {
            name: float(value)
            for name, value in metrics.items()
        }
    }

    (output_dir / "test_metrics.json").write_text(
        json.dumps(evaluation_details, indent=2),
        encoding="utf-8"
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual No", "Actual Yes"],
        columns=["Predicted No", "Predicted Yes"]
    )

    matrix_df.to_csv(output_dir / "confusion_matrix.csv")

    prediction_results = pd.DataFrame({
        "customerID": df.loc[X_test.index, "customerID"],
        "Actual_Churn": y_test,
        "Churn_Score": scores,
        "Predicted_Churn": predictions
    })

    prediction_results.to_csv(
        output_dir / "test_predictions.csv",
        index=False
    )

    coefficients = get_coefficients(model)
    coefficients.to_csv(
        output_dir / "model_coefficients.csv",
        index=False
    )

    plot_churn_distribution(
        y_train,
        output_dir / "churn_distribution.png"
    )

    plot_confusion_matrix(
        matrix,
        THRESHOLD,
        output_dir / "confusion_matrix.png"
    )

    plot_coefficients(
        coefficients,
        output_dir / "model_coefficients.png"
    )

    print("\nGenerating exploratory charts...")

    generate_exploratory_plots(
        df,
        X_train,
        y_train,
        output_dir / "exploratory"
    )

    print("\nReproducing historical experiment charts...")

    run_historical_experiments(
        df,
        output_dir / "experiments"
    )
    print(f"\nResults and charts saved to:\n{output_dir}")


if __name__ == "__main__":
    main()