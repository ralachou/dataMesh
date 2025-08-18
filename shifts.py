import pandas as pd
import numpy as np

import pandas as pd


def process_shift_analysis_from_excel(file_path: str) -> pd.DataFrame:
    """
    Reads an Excel file containing TSKey, Date, Value, abs adjustment,
    computes adjusted values and three methods for 10-day shifts.
    """
    # Load Excel into DataFrame
    df = pd.read_excel(file_path)

    # Ensure proper sorting (newest to oldest)
    df = df.sort_values("Date", ascending=False).reset_index(drop=True)

    # Step 1: Compute Adjusted Value using cumulative abs adjustment
    df["Abs_Adjustment"] = df["abs adjustment"]
    latest_val = df.loc[0, "Value"]
    df["AdjustedValue"] = df["Value"] + df["Abs_Adjustment"].cumsum()

    # Step 2: Compute 1D shifts using Adjusted Value
    df["Raw_1D_Shift"] = df["Value"] - df["Value"].shift(-1)
    df["Adjusted_1D_Shift"] = df["AdjustedValue"] - df["AdjustedValue"].shift(-1)

    # Step 3: Method 1 – AdjustedValue(D10) - AdjustedValue(D1)
    df["Method1_10DShift"] = df["AdjustedValue"] - df["AdjustedValue"].shift(-9)

    # Step 4: Method 2 – Sum of Adjusted 1D shifts (rolling)
    df["Method2_10DShift"] = df["Adjusted_1D_Shift"].rolling(window=10, min_periods=1).sum()

    # Step 5: Method 3 – Sum of Raw 1D shifts + abs adjustment (rolling)
    df["Combined_Shift_Component"] = df["Raw_1D_Shift"] + df["Abs_Adjustment"]
    df["Method3_10DShift"] = df["Combined_Shift_Component"].rolling(window=10, min_periods=1).sum()

    return df[["TSKey", "Date", "Value", "Abs_Adjustment", "AdjustedValue",
               "Raw_1D_Shift", "Adjusted_1D_Shift", "Method1_10DShift",
               "Method2_10DShift", "Method3_10DShift"]]


# Attempt to load and process (assuming the user uploads the Excel next)




# Step 1: Create the dataset (newest to oldest)
dates = pd.date_range(start="2020-12-14", end="2021-01-20")[::-1]
values = [3]*8 + [2]*12 + [1]*13 + [0]*5
df = pd.DataFrame({"Date": dates, "Value": values})

# Step 2: Sort and compute Adjusted Value
df = df.sort_values("Date", ascending=False).reset_index(drop=True)
latest_val = df.loc[0, "Value"]
df["Abs_Adjustment"] = np.where(df["Value"] < latest_val, latest_val - df["Value"], 0)
df["AdjustedValue"] = df["Value"] + df["Abs_Adjustment"]

# Step 3: Compute 1D shifts
df["Raw_1D_Shift"] = df["Value"] - df["Value"].shift(-1)
df["Adjusted_1D_Shift"] = df["Raw_1D_Shift"]
df.loc[df["Abs_Adjustment"] > 0, "Adjusted_1D_Shift"] += df["Abs_Adjustment"]

# Step 4: Method 1 – Shift from AdjustedValue(D10) - AdjustedValue(D1)
df["Method1_10DShift"] = df["AdjustedValue"] - df["AdjustedValue"].shift(-9)

# Step 5: Method 2 – Sum of Adjusted 1D shifts (rolling)
df["Method2_10DShift"] = df["Adjusted_1D_Shift"].rolling(window=10, min_periods=1).sum()

# Step 6: Method 3 – Sum of (Raw 1D + Adjustment)
df["Combined_Shift_Component"] = df["Raw_1D_Shift"] + df["Abs_Adjustment"]
df["Method3_10DShift"] = df["Combined_Shift_Component"].rolling(window=10, min_periods=1).sum()

# Step 7: Show final result
cols_to_show = [
    "Date", "Value", "AdjustedValue", "Raw_1D_Shift", "Adjusted_1D_Shift",
    "Abs_Adjustment", "Method1_10DShift", "Method2_10DShift", "Method3_10DShift"
]
df_result = df[cols_to_show]

# Display top 25 rows from most recent date
print(df_result.head(25).to_string(index=False))
