from pathlib import Path
from dataclasses import dataclass
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col


MIN_PRE_TREATMENT_POINTS = 3
MIN_POST_TREATMENT_POINTS = 3


@dataclass
class RegressionAnalysis:
    """Bayesian regression results for one measurement."""

    name: str
    dates: np.ndarray
    values: np.ndarray
    pre_model: linear_model.BayesianRidge | None
    post_model: linear_model.BayesianRidge | None
    pre_slope_interval: tuple[float, float] | None
    post_slope_interval: tuple[float, float] | None
    post_prediction_lower: np.ndarray | None
    post_prediction_upper: np.ndarray | None
    post_outlier_mask: np.ndarray | None


def _fit_bayesian_regression(
    dates: np.ndarray, values: np.ndarray
) -> linear_model.BayesianRidge:
    """Fit a Bayesian linear regression to numeric dates and measurements."""
    model = linear_model.BayesianRidge()
    model.fit(dates.reshape(-1, 1), values)
    return model


def _slope_interval(
    model: linear_model.BayesianRidge, confidence: float = 0.95
) -> tuple[float, float]:
    """Return a normal approximation to the posterior interval for the slope."""
    standard_error = float(np.sqrt(model.sigma_[0, 0]))
    margin = 1.96 * standard_error if confidence == 0.95 else standard_error
    return float(model.coef_[0] - margin), float(model.coef_[0] + margin)


def _predictive_interval(
    model: linear_model.BayesianRidge, dates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the 95% posterior predictive interval for new observations."""
    design = dates.reshape(-1, 1)
    predictive_variance = (1 / model.alpha_) + np.einsum(
        "ij,jk,ik->i", design, model.sigma_, design
    )
    standard_deviation = 1.96 * np.sqrt(np.maximum(predictive_variance, 0))
    prediction = model.predict(dates.reshape(-1, 1))
    return prediction - standard_deviation, prediction + standard_deviation


def _rows_to_arrays(rows, value_column: str) -> tuple[np.ndarray, np.ndarray]:
    """Convert Spark rows to finite numeric date and measurement arrays."""
    pairs = []
    for row in rows:
        try:
            date = float(row["days_since_start"])
            value = float(row[value_column])
        except (TypeError, ValueError):
            continue
        if np.isfinite(date) and np.isfinite(value):
            pairs.append((date, value))

    if not pairs:
        return np.array([]), np.array([])
    pairs.sort()
    dates, values = zip(*pairs)
    return np.asarray(dates), np.asarray(values)


def analyse_measurements(
    rows, value_column: str, name: str
) -> RegressionAnalysis | None:
    """Analyse one measurement, or return None when there is insufficient data."""
    dates, values = _rows_to_arrays(rows, value_column)
    pre_mask = dates < 0
    post_mask = dates >= 0
    has_pre_treatment_regression = pre_mask.sum() >= MIN_PRE_TREATMENT_POINTS
    has_post_treatment_regression = post_mask.sum() >= MIN_POST_TREATMENT_POINTS
    if not has_pre_treatment_regression and not has_post_treatment_regression:
        return None

    pre_model = (
        _fit_bayesian_regression(dates[pre_mask], values[pre_mask])
        if has_pre_treatment_regression
        else None
    )
    post_model = (
        _fit_bayesian_regression(dates[post_mask], values[post_mask])
        if has_post_treatment_regression
        else None
    )
    lower = upper = outlier_mask = None
    if pre_model is not None and post_mask.any():
        post_dates = dates[post_mask]
        lower, upper = _predictive_interval(pre_model, post_dates)
        outlier_mask = (values[post_mask] < lower) | (values[post_mask] > upper)
    return RegressionAnalysis(
        name=name,
        dates=dates,
        values=values,
        pre_model=pre_model,
        post_model=post_model,
        pre_slope_interval=(
            _slope_interval(pre_model) if pre_model is not None else None
        ),
        post_slope_interval=(
            _slope_interval(post_model) if post_model is not None else None
        ),
        post_prediction_lower=lower,
        post_prediction_upper=upper,
        post_outlier_mask=outlier_mask,
    )


def analyse_patient_history(
    bmi_rows, hba1c_rows
) -> dict[str, RegressionAnalysis]:
    """Run the pre/post Bayesian analysis for BMI and HbA1c."""
    analyses = {}
    for rows, value_column, name in (
        (bmi_rows, "observation_value", "BMI"),
        (hba1c_rows, "numeric_value", "HbA1c"),
    ):
        analysis = analyse_measurements(rows, value_column, name)
        if analysis is not None:
            analyses[name] = analysis
    return analyses


def pull_patient_history(bmi_data: DataFrame, hba1c_data: DataFrame, patient_id: str) -> tuple[DataFrame, DataFrame]:
    """Return a patient's BMI and HbA1c measurements ordered by date."""

    bmi_history = bmi_data.filter(col("pat_id") == patient_id).orderBy("days_since_start").select("days_since_start", "observation_value")
    hba1c_history = hba1c_data.filter(col("pat_id") == patient_id).orderBy("days_since_start").select("days_since_start", "numeric_value")

    return bmi_history, hba1c_history


def visualise_patient_history(
    bmi_history: DataFrame,
    hba1c_history: DataFrame,
    patient_id: str,
    analyses: dict[str, RegressionAnalysis] | None = None,
):
    """Plot a patient's BMI and HbA1c measurements over time."""
    bmi_rows = bmi_history.collect()
    hba1c_rows = hba1c_history.collect()

    bmi_dates = [row["days_since_start"] for row in bmi_rows]
    bmi_values = [float(row["observation_value"]) for row in bmi_rows]
    hba1c_dates = [row["days_since_start"] for row in hba1c_rows]
    hba1c_values = [float(row["numeric_value"]) for row in hba1c_rows]

    figure, bmi_axis = plt.subplots(figsize=(10, 6))
    hba1c_axis = bmi_axis.twinx()

    bmi_line = bmi_axis.plot(
        bmi_dates, bmi_values, marker="o", color="tab:blue", label="BMI"
    )
    hba1c_line = hba1c_axis.plot(
        hba1c_dates, hba1c_values, marker="o", color="tab:orange", label="HbA1c"
    )

    bmi_axis.axvline(x=0, color="tab:red", label="Treatment start")
    axes_by_name = {"BMI": bmi_axis, "HbA1c": hba1c_axis}
    colors_by_name = {"BMI": "tab:blue", "HbA1c": "tab:orange"}
    for name, analysis in (analyses or {}).items():
        axis = axes_by_name[name]
        color = colors_by_name[name]
        if analysis.pre_model is not None:
            pre_dates = np.linspace(
                analysis.dates[analysis.dates < 0].min(), 0, 100
            )
            axis.plot(
                pre_dates,
                analysis.pre_model.predict(pre_dates.reshape(-1, 1)),
                color=color,
                linestyle="--",
                label=f"{name} pre-treatment fit",
            )
        if analysis.post_model is not None:
            post_dates = np.linspace(
                0, analysis.dates[analysis.dates >= 0].max(), 100
            )
            axis.plot(
                post_dates,
                analysis.post_model.predict(post_dates.reshape(-1, 1)),
                color=color,
                linestyle=":",
                label=f"{name} post-treatment fit",
            )
        if analysis.pre_model is not None and analysis.post_prediction_lower is not None:
            post_dates = analysis.dates # changed from analysis.dates[analysis.dates >= 0]
            post_lower, post_upper = _predictive_interval(
                analysis.pre_model, post_dates
            )
            axis.fill_between(
                post_dates,
                post_lower,
                post_upper,
                color=color,
                alpha=0.12,
                label=f"{name} pre-treatment 95% predictive interval",
            )
            post_dates_observed = analysis.dates[analysis.dates >= 0]
            axis.scatter(
                post_dates_observed[analysis.post_outlier_mask],
                analysis.values[analysis.dates >= 0][analysis.post_outlier_mask],
                facecolors="none",
                edgecolors="red",
                s=80,
                label=f"{name} outside pre-treatment 95% interval",
            )

    bmi_axis.set_xlabel("Date")
    bmi_axis.set_ylabel("BMI", color="tab:blue")
    hba1c_axis.set_ylabel("HbA1c (%)", color="tab:orange")
    bmi_axis.set_ylim(10, 50)
    hba1c_axis.set_ylim(0, 15)
    figure.suptitle(f"BMI and HbA1c History for Patient {patient_id}")

    lines, labels = [], []
    for axis in (bmi_axis, hba1c_axis):
        axis_lines, axis_labels = axis.get_legend_handles_labels()
        lines.extend(axis_lines)
        labels.extend(axis_labels)
    bmi_axis.legend(lines, labels, loc="best")
    figure.tight_layout()
    return figure


if __name__ == "__main__":

    data_dir = Path(__file__).resolve().parent
    spark = SparkSession.builder.getOrCreate()


    # cohort_data = spark.read.option("header", True).option("inferSchema", True).csv(str(data_dir / "Cohort Data/cohort_5_final.csv"))
    bmi_data = spark.read.option("header", True).option("inferSchema", True).csv(str(data_dir / "All Readings - BMI + AC1/clean_bmi_2yr_before.csv"))
    hba1c_data = spark.read.option("header", True).option("inferSchema", True).csv(str(data_dir / "All Readings - BMI + AC1/clean_A1C_2yr_before.csv"))


    while True:
        patient_id = input("Enter a patient ID (or 'q' to quit): ")
        if patient_id.lower() == "q":
            break

        BMI_history, HbA1c_history = pull_patient_history(bmi_data, hba1c_data, patient_id)

        if BMI_history.count() == 0 and HbA1c_history.count() == 0:
            print(f"No data found for patient ID: {patient_id}")
            continue

        bmi_rows = BMI_history.collect()
        hba1c_rows = HbA1c_history.collect()
        analyses = analyse_patient_history(bmi_rows, hba1c_rows)
        if not analyses:
            print(
                "Insufficient data for Bayesian analysis "
                f"(need at least {MIN_PRE_TREATMENT_POINTS} pre-treatment or "
                f"{MIN_POST_TREATMENT_POINTS} post-treatment points per measure)."
            )
        else:
            for name, analysis in analyses.items():
                if analysis.pre_model is not None:
                    print(
                        f"{name}: pre-treatment slope 95% interval="
                        f"{analysis.pre_slope_interval}"
                    )
                if analysis.post_model is not None:
                    print(
                        f"{name}: post-treatment slope 95% interval="
                        f"{analysis.post_slope_interval}"
                    )
                if analysis.post_outlier_mask is not None:
                    print(
                        f"{name}: {int(analysis.post_outlier_mask.sum())} "
                        "post-treatment point(s) outside the pre-treatment "
                        "95% predictive interval"
                    )

        figure = visualise_patient_history(
            BMI_history, HbA1c_history, patient_id, analyses
        )
        plt.show(block=True)
