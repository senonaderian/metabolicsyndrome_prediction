import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('robust_normalized_data.csv')

# Count the number of rows before deleting
total_rows_before = df.shape[0]

# Delete rows with empty cells
df = df.dropna()

# Count the number of rows after deleting
total_rows_after = df.shape[0]

# Print the updated DataFrame
print(df)

# Print the total number of deleted rows
deleted_rows = total_rows_before - total_rows_after
print("Total number of deleted rows:", deleted_rows)

# Save the updated DataFrame to a CSV file
df.to_csv('3.csv', index=False)
