from pathlib import Path

import matplotlib.pyplot as plt
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col


def pull_patient_history(patient_id: str) -> tuple[DataFrame, DataFrame]:
    """Return a patient's BMI and HbA1c measurements ordered by date."""
    if not isinstance(patient_id, str) or not patient_id.strip():
        raise ValueError("patient_id must be a non-empty string")

    data_dir = Path(__file__).resolve().parent
    spark = SparkSession.builder.getOrCreate()

    def read_history(filename: str, date_column: str, val_column:str) -> DataFrame:
        return (
            spark.read.option("header", True)
            .option("inferSchema", True)
            .csv(str(data_dir / filename))
            .filter(col("pat_id") == patient_id)
            .orderBy(date_column).select(date_column, val_column)
        )

    bmi_history = read_history("glp1_bmi.csv", "recorded_time", "observation_value")
    hba1c_history = read_history("glp1_a1c.csv", "result_time", "numeric_value")

    return bmi_history, hba1c_history


def visualize_patient_history(bmi_history: DataFrame, hba1c_history: DataFrame, patient_id: str):
    """Plot a patient's BMI and HbA1c measurements over time."""
    bmi_rows = bmi_history.collect()
    hba1c_rows = hba1c_history.collect()

    bmi_dates = [row["recorded_time"] for row in bmi_rows]
    bmi_values = [float(row["observation_value"]) for row in bmi_rows]
    hba1c_dates = [row["result_time"] for row in hba1c_rows]
    hba1c_values = [float(row["numeric_value"]) for row in hba1c_rows]

    figure, bmi_axis = plt.subplots(figsize=(10, 6))
    hba1c_axis = bmi_axis.twinx()

    bmi_line = bmi_axis.plot(
        bmi_dates, bmi_values, marker="o", color="tab:blue", label="BMI"
    )
    hba1c_line = hba1c_axis.plot(
        hba1c_dates, hba1c_values, marker="o", color="tab:red", label="HbA1c"
    )

    bmi_axis.set_xlabel("Date")
    bmi_axis.set_ylabel("BMI", color="tab:blue")
    hba1c_axis.set_ylabel("HbA1c (%)", color="tab:red")
    hba1c_axis.set_ylim(0, 15)
    figure.suptitle(f"BMI and HbA1c History for Patient {patient_id}")

    lines = bmi_line + hba1c_line
    bmi_axis.legend(lines, [line.get_label() for line in lines], loc="best")
    figure.tight_layout()
    return figure


if __name__ == "__main__":

    #example usage
    patient_id = "08da1509db489c7c056fe1ab9245f9621bc33dfa82553a994d63bcbdd0fffb75"
    BMI_history, HbA1c_history = pull_patient_history(patient_id)

    # BMI_history.show()
    # HbA1c_history.show()

    figure = visualize_patient_history(BMI_history, HbA1c_history, patient_id)
    plt.show(block=True)