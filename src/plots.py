from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.patches import Patch
from plotly.subplots import make_subplots
from sklearn.metrics import ConfusionMatrixDisplay


def save_figure(fig, output_path):
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_churn_distribution(y_train, output_path):
    counts = y_train.value_counts().reindex(
        ["No", "Yes"], fill_value=0
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=["#3B82F6", "#EF4444"]
    )

    ax.bar_label(bars, padding=3)
    ax.set_title("Churn Distribution — Training Customers")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of customers")
    ax.margins(y=0.15)

    save_figure(fig, output_path)


def plot_confusion_pair(matrix, title, output_path):
    output_path = Path(output_path)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        matrix,
        display_labels=["No", "Yes"]
    ).plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title(f"{title}\nCustomer Counts")
    save_figure(fig, output_path)

    row_totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        percentages,
        display_labels=["No", "Yes"]
    ).plot(
        ax=ax,
        cmap="Blues",
        values_format=".1%",
        colorbar=False,
        im_kw={"vmin": 0, "vmax": 1}
    )

    ax.set_title(f"{title}\nPercent Within Each Actual Class")

    percentage_path = output_path.with_name(
        f"{output_path.stem}_percent{output_path.suffix}"
    )
    save_figure(fig, percentage_path)


def plot_confusion_matrix(matrix, threshold, output_path):
    plot_confusion_pair(
        matrix,
        f"Final Model — Test Set — Threshold {threshold:.2f}",
        output_path
    )


def plot_coefficients(coefficients, output_path):
    strongest = (
        coefficients.assign(
            magnitude=coefficients["Coefficient"].abs()
        )
        .nlargest(20, "magnitude")
        .sort_values("Coefficient")
    )

    colors = [
        "#EF4444" if value > 0 else "#3B82F6"
        for value in strongest["Coefficient"]
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        strongest["Feature"],
        strongest["Coefficient"],
        color=colors
    )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("20 Largest Logistic Regression Coefficients")
    ax.set_xlabel(
        "Coefficient — positive values increase the churn score"
    )

    save_figure(fig, output_path)


def plot_gender_and_churn(df, output_dir):
    gender_counts = df["gender"].value_counts().reindex(
        ["Female", "Male"], fill_value=0
    )
    churn_counts = df["Churn"].value_counts().reindex(
        ["No", "Yes"], fill_value=0
    )

    # Static version for opening in VS Code and displaying on GitHub.
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].pie(
        gender_counts.values,
        labels=gender_counts.index,
        autopct="%1.1f%%",
        colors=["#A5B4FC", "#6366F1"],
        startangle=90,
        wedgeprops={"width": 0.45, "edgecolor": "white"}
    )
    axes[0].set_title("Gender Distribution")

    axes[1].pie(
        churn_counts.values,
        labels=churn_counts.index,
        autopct="%1.1f%%",
        colors=["#3B82F6", "#EF4444"],
        startangle=90,
        wedgeprops={"width": 0.45, "edgecolor": "white"}
    )
    axes[1].set_title("Churn Distribution")

    fig.suptitle("Telco Customers — Gender and Churn")
    save_figure(fig, output_dir / "gender_and_churn.png")

    # Interactive version matching the notebook's Plotly chart.
    interactive = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=["Gender distribution", "Churn distribution"]
    )

    interactive.add_trace(
        go.Pie(
            labels=gender_counts.index,
            values=gender_counts.values,
            marker_colors=["#A5B4FC", "#6366F1"],
            name="Gender"
        ),
        row=1,
        col=1
    )

    interactive.add_trace(
        go.Pie(
            labels=churn_counts.index,
            values=churn_counts.values,
            marker_colors=["#3B82F6", "#EF4444"],
            name="Churn"
        ),
        row=1,
        col=2
    )

    interactive.update_traces(
        hole=0.45,
        textinfo="label+percent",
        hovertemplate=(
            "%{label}: %{value} customers "
            "(%{percent})<extra></extra>"
        )
    )

    interactive.update_layout(
        title_text="Telco Customers — Gender and Churn",
        template="plotly_white",
        height=450,
        showlegend=False
    )

    interactive.write_html(
        str(output_dir / "gender_and_churn.html"),
        include_plotlyjs=True,
        full_html=True
    )


def plot_nested_gender_churn(df, output_path):
    counts = pd.crosstab(df["Churn"], df["gender"]).reindex(
        index=["Yes", "No"],
        columns=["Female", "Male"],
        fill_value=0
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.pie(
        counts.sum(axis=1).to_numpy(),
        radius=1,
        labels=["Churn: Yes", "Churn: No"],
        colors=["#EF4444", "#3B82F6"],
        startangle=90,
        autopct="%1.1f%%",
        pctdistance=0.85,
        wedgeprops={"width": 0.3, "edgecolor": "white"}
    )

    ax.pie(
        counts.to_numpy().flatten(),
        radius=0.7,
        colors=["#A5B4FC", "#6366F1"] * 2,
        startangle=90,
        wedgeprops={"width": 0.3, "edgecolor": "white"}
    )

    ax.legend(
        handles=[
            Patch(facecolor="#A5B4FC", label="Female"),
            Patch(facecolor="#6366F1", label="Male")
        ],
        title="Inner ring",
        loc="lower right"
    )

    ax.set_title("Churn Distribution by Gender", pad=20)
    ax.set_aspect("equal")
    save_figure(fig, output_path)


def generate_exploratory_plots(df, X_train, y_train, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # These charts use the full dataset, matching the notebook.
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(
        data=df,
        x="Churn",
        order=["No", "Yes"],
        color="#6366F1",
        ax=ax
    )
    ax.set_title("Customer Churn Distribution")
    save_figure(fig, output_dir / "churn_distribution_full_dataset.png")

    contract_rates = pd.crosstab(
        df["Contract"],
        df["Churn"],
        normalize="index"
    ).reindex(columns=["No", "Yes"], fill_value=0) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    contract_rates.plot(
        kind="bar",
        stacked=True,
        color=["#3B82F6", "#EF4444"],
        ax=ax
    )
    ax.set_title("Churn by Contract Type")
    ax.set_ylabel("Percentage")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=15)
    save_figure(fig, output_dir / "churn_by_contract.png")

    for column, title, filename in [
        (
            "tenure",
            "Customer Tenure by Churn Status",
            "tenure_by_churn.png"
        ),
        (
            "MonthlyCharges",
            "Monthly Charges by Churn Status",
            "monthly_charges_by_churn.png"
        )
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(
            data=df,
            x="Churn",
            y=column,
            order=["No", "Yes"],
            ax=ax
        )
        ax.set_title(title)
        save_figure(fig, output_dir / filename)

    plot_gender_and_churn(df, output_dir)
    plot_nested_gender_churn(
        df,
        output_dir / "churn_by_gender_nested.png"
    )

    # These charts use original training customers, matching the notebook.
    analysis = X_train.copy()
    analysis["Churn"] = y_train
    analysis["ChurnBinary"] = (y_train == "Yes").astype(int)

    analysis["TenureGroup"] = pd.cut(
        analysis["tenure"],
        bins=[-1, 6, 12, 24, 36, 48, 60, 72],
        labels=["0-6", "7-12", "13-24", "25-36",
                "37-48", "49-60", "61-72"]
    )

    tenure_rates = analysis.groupby(
        "TenureGroup",
        observed=True
    )["ChurnBinary"].mean() * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        tenure_rates.index.astype(str),
        tenure_rates.values,
        color="#6366F1"
    )
    ax.set_title("Churn Rate by Customer Tenure")
    ax.set_xlabel("Customer Tenure (Months)")
    ax.set_ylabel("Churn Rate (%)")
    save_figure(fig, output_dir / "churn_rate_by_tenure.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(x=analysis["tenure"], ax=axes[0])
    axes[0].set_title("Tenure — Outlier Detection")

    sns.boxplot(x=analysis["MonthlyCharges"], ax=axes[1])
    axes[1].set_title("Monthly Charges — Outlier Detection")
    save_figure(fig, output_dir / "outlier_boxplots.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=analysis,
        x="tenure",
        y="MonthlyCharges",
        hue="Churn",
        palette={"Yes": "red", "No": "blue"},
        alpha=0.5,
        ax=ax
    )
    ax.set_title("Tenure vs Monthly Charges by Churn")
    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Monthly Charges")
    save_figure(fig, output_dir / "tenure_vs_monthly_charges.png")

    analysis["TenureGroup"] = pd.cut(
        analysis["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6", "7-12", "13-24", "25-48", "49-72"]
    )

    analysis["ChargeGroup"] = pd.cut(
        analysis["MonthlyCharges"],
        bins=[0, 40, 70, 90, 120],
        labels=["Low", "Medium", "High", "Very High"]
    )

    heatmap = analysis.pivot_table(
        index="TenureGroup",
        columns="ChargeGroup",
        values="ChurnBinary",
        aggfunc="mean",
        observed=True
    ) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        heatmap,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        ax=ax
    )
    ax.set_title("Churn Rate (%) by Tenure and Monthly Charges")
    ax.set_xlabel("Monthly Charge Group")
    ax.set_ylabel("Tenure Group")
    save_figure(fig, output_dir / "tenure_charge_heatmap.png")