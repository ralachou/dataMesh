import pandas as pd

def process_proxy_data_v6(file_path):
    # Load the 'proxy' sheet
    proxy_df = pd.read_excel(file_path, sheet_name='proxy')

    # Identify and remove bad data (AppName = SVAR and StressName = GVAR)
    proxy_df = proxy_df[~((proxy_df['appName'] == 'SVAR') & (proxy_df['stressName'] == 'GVAR'))]

    # Resolve duplicates by keeping the first non-null value
    proxy_df = proxy_df.sort_values(by=['businessDate', 'driverName', 'stressName'])
    proxy_df = proxy_df.drop_duplicates(subset=['driverName', 'stressName'], keep='first')

    # Pivot the data to transform stressName values into columns
    transformed_df = proxy_df.pivot(index=['businessDate', 'driverName'], 
                                    columns='stressName', values='proxyDriver').reset_index()

    # Rename columns for clarity
    transformed_df = transformed_df.rename(columns={'Stress_1': 'Stress1', 'Stress_2': 'Stress2', 'GVAR': 'GVAR'})

    # Replace NaN values with "original"
    transformed_df.fillna('original', inplace=True)

    # Identify drivers missing either Stress_1 or Stress_2 (i.e., appearing less than 3 times in total)
    driver_stress_counts = proxy_df.groupby('driverName')['stressName'].nunique()
    drivers_missing_stress = driver_stress_counts[driver_stress_counts < 3].index.tolist()

    # Apply "not called" for missing Stress1 or Stress2
    for driver in drivers_missing_stress:
        if driver in transformed_df['driverName'].values:
            if 'Stress1' in transformed_df.columns:
                transformed_df.loc[(transformed_df['driverName'] == driver) & 
                                  (transformed_df['Stress1'] == 'original'), 'Stress1'] = 'not called'
            if 'Stress2' in transformed_df.columns:
                transformed_df.loc[(transformed_df['driverName'] == driver) & 
                                  (transformed_df['Stress2'] == 'original'), 'Stress2'] = 'not called'
    
    # Add a new column to flag issues where the number of rows is not 3
    transformed_df['IssueFlag'] = 'No Issue'
    for driver in driver_stress_counts.index:
        if driver_stress_counts[driver] != 3:
            transformed_df.loc[transformed_df['driverName'] == driver, 'IssueFlag'] = f'Row count issue: {driver_stress_counts[driver]}'
    
    return transformed_df

# Example usage:
# transformed_proxy_data = process_proxy_data_v6("/mnt/data/position.xlsx")
