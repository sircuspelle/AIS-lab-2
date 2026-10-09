import pandas as pd

df = pd.read_csv("dataset.csv")
df = df.dropna(subset=['defaultTDP'])

cols_to_check = ['platform', 'family', 'cpuSocket', 'line']

for col in cols_to_check:
    print(f"--- Column: {col} ---")
    print(f"Number of unique values: {df[col].nunique()}")
    print("Top 5 values by count:")
    print(df[col].value_counts().head(8))
    print("\n")
