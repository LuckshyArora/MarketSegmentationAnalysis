import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
import statsmodels.api as sm

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

# Standardize features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# k-Means Clustering
kmeans = KMeans(n_clusters=4, n_init=10, random_state=1234)
df["cluster_kmeans"] = kmeans.fit_predict(df[features])

# Evaluate k-Means Clustering
silhouette_kmeans = silhouette_score(df[features], df["cluster_kmeans"])
print(f"Silhouette Score for k-Means: {silhouette_kmeans}")

# Mixture of Gaussians (GMM) Clustering
gmm = GaussianMixture(n_components=4, n_init=10, random_state=1234)
df["cluster_gmm"] = gmm.fit_predict(df[features])

# Evaluate GMM Clustering
silhouette_gmm = silhouette_score(df[features], df["cluster_gmm"])
print(f"Silhouette Score for GMM: {silhouette_gmm}")

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
df["cluster_dbscan"] = dbscan.fit_predict(df[features])

# Evaluate DBSCAN Clustering
silhouette_dbscan = silhouette_score(df[features], df["cluster_dbscan"])
print(f"Silhouette Score for DBSCAN: {silhouette_dbscan}")

# Linear Regression
X = df[features]  # Independent variables
y = df["Like.n"]  # Dependent variable
X = sm.add_constant(X)  # Add intercept
model = sm.OLS(y, X).fit()
print(model.summary())

# Evaluate Regression Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
model_train = sm.OLS(y_train, X_train).fit()
y_pred = model_train.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Data Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Scatter plot for clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='Like.n', hue='cluster_kmeans', palette='viridis')
plt.title('k-Means Clustering')
plt.show()

# Additional visualizations as needed...

# Hyperparameter Tuning for k-Means
param_grid_kmeans = {'n_clusters': [2, 3, 4, 5, 6]}
grid_kmeans = GridSearchCV(KMeans(n_init=10, random_state=1234), param_grid_kmeans, cv=5)
grid_kmeans.fit(df[features])
print(f"Best Params for k-Means: {grid_kmeans.best_params_}")

# Hyperparameter Tuning for GMM
param_grid_gmm = {'n_components': [2, 3, 4, 5, 6]}
grid_gmm = GridSearchCV(GaussianMixture(n_init=10, random_state=1234), param_grid_gmm, cv=5)
grid_gmm.fit(df[features])
print(f"Best Params for GMM: {grid_gmm.best_params_}")
