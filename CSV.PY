import pandas as pd

# Sample data based on expected model features
data = {
    'feature1': [1.2, 2.3, 3.4],
    'feature2': [3.4, 4.5, 5.6],
    'feature3': [5.6, 6.7, 7.8]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save as CSV and XLSX
df.to_csv('sample_input.csv', index=False)
df.to_excel('sample_input.xlsx', index=False)
