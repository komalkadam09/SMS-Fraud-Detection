import pandas as pd

# Load dataset
data = pd.read_csv("data/spam.csv", encoding="latin-1")

print("Original Shape:")
print(data.shape)

# Remove unnecessary columns
data = data.drop(columns=[
    "Unnamed: 2",
    "Unnamed: 3",
    "Unnamed: 4"
])

print("\nShape After Removing Columns:")
print(data.shape)

# Rename columns
data.columns = ["label", "message"]

print("\nColumn Names:")
print(data.columns)

print("\nFirst 5 Rows:")
print(data.head())