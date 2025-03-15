import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder

# Load Data
df = pd.read_csv("/mnt/data/mcdonalds.csv")

# Convert categorical "Yes"/"No" responses to numeric (1/0)
categorical_cols = ["yummy", "convenient", "spicy", "fattening", "greasy", "fast", "cheap", "tasty", "expensive", "healthy", "disgusting"]
df[categorical_cols] = df[categorical_cols].apply(lambda x: x.map({"Yes": 1, "No": 0}))

# Convert Like column to numeric
df["Like.n"] = df["Like"].replace({"+5": 5, "+4": 4, "+3": 3, "+2": 2, "+1": 1, "0": 0, "-1": -1, "-2": -2, "-3": -3, "-4": -4, "-5": -5}).astype(int)

# Encode categorical columns like VisitFrequency and Gender
label_enc = LabelEncoder()
df["VisitFrequency"] = label_enc.fit_transform(df["VisitFrequency"])
df["Gender"] = label_enc.fit_transform(df["Gender"])

# Select features for clustering and modeling
features = categorical_cols + ["Age", "VisitFrequency", "Gender"]

# k-Means Clustering
kmeans = KMeans(n_clusters=4, n_init=10, random_state=1234)
df["cluster_kmeans"] = kmeans.fit_predict(df[features])

# Mixture of Gaussians (GMM) Clustering
gmm = GaussianMixture(n_components=4, n_init=10, random_state=1234)
df["cluster_gmm"] = gmm.fit_predict(df[features])

# Linear Regression
X = df[features]  # Independent variables
y = df["Like.n"]  # Dependent variable
X = sm.add_constant(X)  # Add intercept
model = sm.OLS(y, X).fit()
print(model.summary())
