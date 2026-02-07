import pandas as pd

CSV_PATH = "gestures.csv"   # change path if needed

df = pd.read_csv(CSV_PATH)

print("CSV file:", CSV_PATH)
print("-" * 50)

print("Total rows:", df.shape[0])
print("Total columns:", df.shape[1])

print("\nFirst 10 column names:")
print(df.columns[:10].tolist())

print("\nLast 5 column names:")
print(df.columns[-5:].tolist())

print("\nLabel column name:", df.columns[-1])

# Count feature columns (everything except label)
feature_cols = df.shape[1] - 1
print("\nFeature columns:", feature_cols)

# Check unique labels
print("\nUnique labels:")
print(df[df.columns[-1]].unique())
