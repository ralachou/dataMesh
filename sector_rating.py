import pandas as pd


import pandas as pd

def get_top_values(time_series):
    """
    This function takes a time series for a curve and returns the top 1 and top 2 
    max values with their respective dates.
    
    :param time_series: pandas DataFrame with 'Date' and 'Value' columns
    :return: Tuple containing:
             - Top 1 max value with its date
             - Top 2 max values with their dates
    """
    # Ensure 'Date' is datetime format and 'Value' is numeric
    time_series['Date'] = pd.to_datetime(time_series['Date'])
    time_series['Value'] = pd.to_numeric(time_series['Value'], errors='coerce')

    # Sort by value in descending order
    sorted_series = time_series.sort_values(by='Value', ascending=False)

    # Top 1: Max value with its date
    top_1 = sorted_series.iloc[0]

    # Top 2: Two largest values with their dates
    top_2 = sorted_series.head(2)

    # Return top 1 and top 2
    return top_1[['Date', 'Value']], top_2[['Date', 'Value']]

# Example of usage:
# Sample DataFrame for time series
data = {'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
        'Value': [100, 250, 150, 300]}

df = pd.DataFrame(data)

# Get top values
top_1, top_2 = get_top_values(df)

print("Top 1 max value and date:", top_1)
print("Top 2 max values and dates:", top_2)





def compute_shifts(sector_curve, rating_curve):
    """
    This function computes the absolute and percentage shifts between sector curve 
    and rating curve for each row.
    
    :param sector_curve: pandas DataFrame with 'Date' and 'Value' for sector curve
    :param rating_curve: pandas DataFrame with 'Date' and 'Value' for rating curve
    :return: pandas DataFrame with the absolute and percentage shifts for each date
    """
    # Ensure 'Date' is datetime format and 'Value' is numeric for both curves
    sector_curve['Date'] = pd.to_datetime(sector_curve['Date'])
    sector_curve['Value'] = pd.to_numeric(sector_curve['Value'], errors='coerce')

    rating_curve['Date'] = pd.to_datetime(rating_curve['Date'])
    rating_curve['Value'] = pd.to_numeric(rating_curve['Value'], errors='coerce')

    # Merge both DataFrames on Date
    merged_df = pd.merge(sector_curve, rating_curve, on='Date', suffixes=('_sector', '_rating'))

    # Calculate Absolute Shift (rating - sector)
    merged_df['Absolute_Shift'] = merged_df['Value_rating'] - merged_df['Value_sector']

    # Calculate Percentage Shift ((rating - sector) / sector * 100)
    merged_df['Percentage_Shift'] = ((merged_df['Value_rating'] - merged_df['Value_sector']) / merged_df['Value_sector']) * 100

    return merged_df[['Date', 'Value_sector', 'Value_rating', 'Absolute_Shift', 'Percentage_Shift']]

# Example usage:
# Sample sector curve and rating curve data
sector_data = {'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
               'Value': [100, 250, 150, 300]}

rating_data = {'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
               'Value': [110, 240, 160, 310]}

sector_df = pd.DataFrame(sector_data)
rating_df = pd.DataFrame(rating_data)

# Compute the shifts
shifts_df = compute_shifts(sector_df, rating_df)

# Display the result
print(shifts_df)


















import pandas as pd
import matplotlib.pyplot as plt

def compute_shifts(sector_curve, rating_curve):
    """
    This function computes the absolute and percentage shifts between sector curve 
    and rating curve for each row.
    
    :param sector_curve: pandas DataFrame with 'Date' and 'Value' for sector curve
    :param rating_curve: pandas DataFrame with 'Date' and 'Value' for rating curve
    :return: pandas DataFrame with the absolute and percentage shifts for each date
    """
    # Ensure 'Date' is datetime format and 'Value' is numeric for both curves
    sector_curve['Date'] = pd.to_datetime(sector_curve['Date'])
    sector_curve['Value'] = pd.to_numeric(sector_curve['Value'], errors='coerce')

    rating_curve['Date'] = pd.to_datetime(rating_curve['Date'])
    rating_curve['Value'] = pd.to_numeric(rating_curve['Value'], errors='coerce')

    # Merge both DataFrames on Date
    merged_df = pd.merge(sector_curve, rating_curve, on='Date', suffixes=('_sector', '_rating'))

    # Calculate Absolute Shift (rating - sector)
    merged_df['Absolute_Shift'] = merged_df['Value_rating'] - merged_df['Value_sector']

    # Calculate Percentage Shift ((rating - sector) / sector * 100)
    merged_df['Percentage_Shift'] = ((merged_df['Value_rating'] - merged_df['Value_sector']) / merged_df['Value_sector']) * 100

    return merged_df[['Date', 'Value_sector', 'Value_rating', 'Absolute_Shift', 'Percentage_Shift']]

def plot_shifts(shifts_df):
    """
    This function plots the comparison between sector and rating curve values,
    as well as the absolute and percentage shifts.
    
    :param shifts_df: pandas DataFrame with the sector, rating, absolute and percentage shifts
    """
    # Create figure and axes
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot the values of sector and rating curves
    ax[0].plot(shifts_df['Date'], shifts_df['Value_sector'], label='Sector Curve', color='blue', marker='o')
    ax[0].plot(shifts_df['Date'], shifts_df['Value_rating'], label='Rating Curve', color='red', marker='x')
    ax[0].set_title('Sector vs Rating Curve')
    ax[0].set_ylabel('Value')
    ax[0].legend(loc='upper left')

    # Plot the Absolute Shift
    ax[1].plot(shifts_df['Date'], shifts_df['Absolute_Shift'], label='Absolute Shift', color='green', marker='s')
    ax[1].set_title('Absolute Shift (Rating - Sector)')
    ax[1].set_ylabel('Shift Value')
    ax[1].set_xlabel('Date')
    ax[1].legend(loc='upper left')

    # Plot the Percentage Shift
    fig, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(shifts_df['Date'], shifts_df['Percentage_Shift'], label='Percentage Shift', color='purple', marker='d')
    ax2.set_title('Percentage Shift (Rating - Sector)')
    ax2.set_ylabel('Percentage Shift')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

# Sample sector curve and rating curve data
sector_data = {'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
               'Value': [100, 250, 150, 300]}

rating_data = {'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
               'Value': [110, 240, 160, 310]}

sector_df = pd.DataFrame(sector_data)
rating_df = pd.DataFrame(rating_data)

# Compute the shifts
shifts_df = compute_shifts(sector_df, rating_df)

# Plot the results
plot_shifts(shifts_df)

