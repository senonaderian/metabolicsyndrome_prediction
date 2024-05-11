import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('robust_normalized_data.csv')

# Get the number of rows with empty cells
total_empty_rows = df.isnull().any(axis=1).sum()

# Print the total number of rows with empty cells
print("Total number of rows with empty cells:", total_empty_rows)
