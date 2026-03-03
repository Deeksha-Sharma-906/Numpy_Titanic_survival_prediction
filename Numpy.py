import numpy as np
import pandas as pd

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("train.csv")

# Convert required columns to NumPy
age = df["Age"].to_numpy()
fare = df["Fare"].to_numpy()
pclass = df["Pclass"].to_numpy()
survived = df["Survived"].to_numpy()
sibsp = df["SibSp"].to_numpy()
parch = df["Parch"].to_numpy()

# Convert Sex to numeric (male=0, female=1)
sex = np.where(df["Sex"] == "female", 1, 0)

# ===============================
# EASY
# ===============================

print("Mean Age:", np.nanmean(age))
print("Min Fare:", np.min(fare))
print("Max Fare:", np.max(fare))
print("Survivors Count:", np.sum(survived == 1))
print("Passengers per Class:", np.bincount(pclass)[1:])
print("Age Std Dev:", np.nanstd(age))

# Fill missing Age with mean
age_filled = np.where(np.isnan(age), np.nanmean(age), age)

# ===============================
# MEDIUM
# ===============================

print("Avg Age of Survivors:", np.mean(age_filled[survived == 1]))
print("Avg Fare 1st Class:", np.mean(fare[pclass == 1]))
print("Female Survivors:", np.sum((sex == 1) & (survived == 1)))

print("Fare > 100:", fare[fare > 100])

survival_rate = np.sum(survived == 1) / len(survived)
print("Survival Rate:", survival_rate)

# Correlation Age & Fare
corr = np.corrcoef(age_filled, fare)[0, 1]
print("Correlation Age-Fare:", corr)

# Family Size
family_size = sibsp + parch
print("Avg Family Size Survivors:",
      np.mean(family_size[survived == 1]))
print("Avg Family Size Non-Survivors:",
      np.mean(family_size[survived == 0]))

# Min-Max Normalization Fare
fare_norm = (fare - np.min(fare)) / (np.max(fare) - np.min(fare))

# Z-score Standardization Age
age_std = (age_filled - np.mean(age_filled)) / np.std(age_filled)

# ===============================
# TOUGH
# ===============================

# 21 Covariance Matrix
X = np.column_stack((age_filled, fare, pclass))
cov_matrix = np.cov(X.T)
print("Covariance Matrix:\n", cov_matrix)

# 22 Eigen Decomposition
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
print("Eigenvalues:\n", eig_vals)

# 23 Logistic Regression Prediction (Manual)
X_lr = np.column_stack((np.ones(len(age_filled)), age_std, fare_norm, pclass))
weights = np.random.randn(X_lr.shape[1])

z = np.dot(X_lr, weights)
sigmoid = 1 / (1 + np.exp(-z))

# 24 Vectorized Prediction
predictions = sigmoid >= 0.5

# 25 Confusion Matrix
TP = np.sum((predictions == 1) & (survived == 1))
TN = np.sum((predictions == 0) & (survived == 0))
FP = np.sum((predictions == 1) & (survived == 0))
FN = np.sum((predictions == 0) & (survived == 1))

conf_matrix = np.array([[TP, FP],
                        [FN, TN]])

print("Confusion Matrix:\n", conf_matrix)

# 26 Cross Entropy Loss
epsilon = 1e-9
loss = -np.mean(
    survived * np.log(sigmoid + epsilon) +
    (1 - survived) * np.log(1 - sigmoid + epsilon)
)
print("Cross Entropy Loss:", loss)

# 27 One Step Gradient Descent
learning_rate = 0.01
gradient = np.dot(X_lr.T, (sigmoid - survived)) / len(survived)
weights = weights - learning_rate * gradient

# 28 Manual Train-Test Split
split = int(0.8 * len(X_lr))
X_train, X_test = X_lr[:split], X_lr[split:]
y_train, y_test = survived[:split], survived[split:]

# 29 IQR Outlier Detection Fare
Q1 = np.percentile(fare, 25)
Q3 = np.percentile(fare, 75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = fare[(fare < lower) | (fare > upper)]
print("Fare Outliers:", outliers)

# 30 Mini PCA
# Center Data
X_centered = X - np.mean(X, axis=0)

# Covariance
cov_pca = np.cov(X_centered.T)

# Eigen
eig_vals_pca, eig_vecs_pca = np.linalg.eig(cov_pca)

# Sort Eigenvalues
idx = np.argsort(eig_vals_pca)[::-1]
eig_vecs_pca = eig_vecs_pca[:, idx]

# Project onto first 2 components
X_pca = np.dot(X_centered, eig_vecs_pca[:, :2])

print("PCA Projection Shape:", X_pca.shape)