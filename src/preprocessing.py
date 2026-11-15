import pandas as pd
import numpy as np

# Step 1: Load the dataset
data = pd.read_csv("dataset/ai4i2020.csv")

# Step 2: Display the first five records
print("\n========== FIRST FIVE ROWS ==========")
print(data.head())

# Step 3: Display dataset information
print("\n========== DATASET INFORMATION ==========")
data.info()

# Step 4: Display the number of rows and columns
print("\n========== DATASET SHAPE ==========")
print(data.shape)

# Step 5: Check for missing values
print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())

# Step 6: Check duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print(data.duplicated().sum())

# Step 7: Display statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(data.describe())