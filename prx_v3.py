import pandas as pd
import openpyxl

def process_proxy_data_v7(file_path):
    # Load the Excel file
    xls = pd.ExcelFile(file_path)
    
    # Load the 'proxy' sheet
    proxy_df = pd.read_excel(xls, sheet_name='proxy')
    
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
    
    # Save transformed data to a new sheet "transformedProxy"
    with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
        transformed_df.to_excel(writer, sheet_name='transformedProxy', index=False)
    
    return transformed_df

def merge_position_with_proxy(file_path):
    # Load the Excel file
    xls = pd.ExcelFile(file_path)
    
    # Load the 'positionData' and 'transformedProxy' sheets
    position_data_df = pd.read_excel(xls, sheet_name='positionData')
    transformed_proxy_df = pd.read_excel(xls, sheet_name='transformedProxy')
    
    # Perform left join with 'positionData' on 'driverName'
    merged_df = position_data_df.merge(transformed_proxy_df, on='driverName', how='left')
    
    # Save merged data into a new sheet "MergedData" while keeping raw data untouched
    with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
        merged_df.to_excel(writer, sheet_name='MergedData', index=False)
    
    return merged_df

def analyze_merged_data(merged_df, file_path):
    summary_levels = {
        'marsProductType': 'mars_productLevel',
        'mroLevel9': 'mro_level9',
        'mrolevel11': 'mro_level11'
    }
    
    total_positions = len(merged_df)
    total_notional = merged_df['notional'].sum()
    total_spread01 = merged_df['Spread01'].sum()
    
    summary_results = []
    
    with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
        for level, group_name in summary_levels.items():
            summary_df = merged_df.groupby(['driverName', level]).agg(
                counts=('positionID', 'count'),
                MV=('marketValueUsd', 'sum'),
                notional_sum=('notional', 'sum'),
                Spread01_sum=('Spread01', 'sum')
            ).reset_index()
            
            summary_df['%count'] = (summary_df['counts'] / total_positions) * 100
            summary_df['%notional_sum'] = (summary_df['notional_sum'] / total_notional) * 100
            summary_df['%Spread01_sum'] = (summary_df['Spread01_sum'] / total_spread01) * 100
            
            summary_df.insert(1, 'groupby', group_name + '_' + summary_df[level].astype(str))
            summary_df.drop(columns=[level], inplace=True)
            
            sheet_name = f'Summary_{group_name}'
            summary_df.to_excel(writer, sheet_name=sheet_name, index=False)
            summary_results.append(sheet_name)
    
    return summary_results

# Example usage:
# transformed_proxy_data = process_proxy_data_v7("/mnt/data/position.xlsx")
# merged_data = merge_position_with_proxy("/mnt/data/position.xlsx")
# summary_sheets = analyze_merged_data(merged_data, "/mnt/data/position.xlsx")
