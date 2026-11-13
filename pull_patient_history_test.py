from pathlib import Path

import matplotlib.pyplot as plt
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col



def pull_patient_history(bmi_data: DataFrame, hba1c_data: DataFrame, patient_id: str) -> tuple[DataFrame, DataFrame]:
    """Return a patient's BMI and HbA1c measurements ordered by date."""

    bmi_history = bmi_data.filter(col("pat_id") == patient_id).orderBy("days_since_start").select("days_since_start", "observation_value")
    hba1c_history = hba1c_data.filter(col("pat_id") == patient_id).orderBy("days_since_start").select("days_since_start", "numeric_value")

    return bmi_history, hba1c_history


def visualize_patient_history(bmi_history: DataFrame, hba1c_history: DataFrame, patient_id: str):
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

    plt.axvline(x=0, color='tab:red')
    bmi_axis.set_xlabel("Date")
    bmi_axis.set_ylabel("BMI", color="tab:blue")
    hba1c_axis.set_ylabel("HbA1c (%)", color="tab:orange")
    bmi_axis.set_ylim(10, 50)
    hba1c_axis.set_ylim(0, 15)
    figure.suptitle(f"BMI and HbA1c History for Patient {patient_id}")

    lines = bmi_line + hba1c_line
    bmi_axis.legend(lines, [line.get_label() for line in lines], loc="best")
    figure.tight_layout()
    return figure


if __name__ == "__main__":

    data_dir = Path(__file__).resolve().parent
    spark = SparkSession.builder.getOrCreate()


    cohort_data = spark.read.option("header", True).option("inferSchema", True).csv(str(data_dir / "Cohort Data/cohort_5_final.csv"))
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

        figure = visualize_patient_history(BMI_history, HbA1c_history, patient_id)
        plt.show(block=True)
