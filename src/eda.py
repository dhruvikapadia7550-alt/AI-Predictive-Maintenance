import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("dataset/ai4i2020.csv")

# Set visualization style
sns.set_style("whitegrid")


# ==========================================
# GRAPH 1: MACHINE FAILURE DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))

sns.countplot(x="Machine failure", data=data)

plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure (0 = No Failure, 1 = Failure)")
plt.ylabel("Number of Machines")

plt.show()


# ==========================================
# GRAPH 2: MACHINE FAILURE PERCENTAGE
# ==========================================

failure_counts = data["Machine failure"].value_counts()

plt.figure(figsize=(7, 7))

plt.pie(
    failure_counts,
    labels=["No Failure", "Failure"],
    autopct="%1.2f%%",
    startangle=90
)

plt.title("Machine Failure Percentage")

plt.show()


# ==========================================
# GRAPH 3: CORRELATION HEATMAP
# ==========================================

numerical_data = data.select_dtypes(include="number")

plt.figure(figsize=(12, 8))

sns.heatmap(
    numerical_data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Machine Parameters")

plt.show()


# ==========================================
# GRAPH 4: TORQUE VS MACHINE FAILURE
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="Machine failure",
    y="Torque [Nm]",
    data=data
)

plt.title("Torque Distribution Based on Machine Failure")
plt.xlabel("Machine Failure (0 = No Failure, 1 = Failure)")
plt.ylabel("Torque [Nm]")

plt.show()


# ==========================================
# GRAPH 5: TOOL WEAR VS MACHINE FAILURE
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="Machine failure",
    y="Tool wear [min]",
    data=data
)

plt.title("Tool Wear Based on Machine Failure")
plt.xlabel("Machine Failure (0 = No Failure, 1 = Failure)")
plt.ylabel("Tool Wear [min]")

plt.show()