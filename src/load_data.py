import pandas as pd 
data= pd.read_csv("data/spam.csv",encoding="latin-1")

print("Dataset Shape:")
print(data.shape)

print("\nFirst 5 Rows:")
print(data.head())

print("\nColumn Names:")
print(data.columns)