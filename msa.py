import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
import statsmodels.api as sm


def load_data(file_path):
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None


def preprocess_data(df):
    """Preprocess the data by encoding categorical variables and standardizing features."""
    categorical_cols = ["yummy", "convenient", "spicy", "fattening", "greasy", "fast", "cheap", "tasty", "expensive", "healthy", "disgusting"]
    df[categorical_cols] = df[categorical_cols].apply(lambda x: x.map({"Yes": 1, "No": 0}))

    df["Like.n"] = df["Like"].replace({"+5": 5, "+4": 4, "+3": 3, "+2": 2, "+1": 1, "0": 0, "-1": -1, "-2": -2, "-3": -3, "-4": -4, "-5": -5}).astype(int)

    label_enc = LabelEncoder()
    df["VisitFrequency"] = label_enc.fit_transform(df["VisitFrequency"])
    df["Gender"] = label_enc.fit_transform(df["Gender"])

    features = categorical_cols + ["Age", "VisitFrequency", "Gender"]

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    return df, features


def apply_pca(df, features, n_components=2):
    """Apply PCA for dimensionality reduction."""
    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df[features])
    return df_pca


def perform_clustering(df, features):
    """Perform clustering using k-Means, GMM, and DBSCAN."""
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=1234)
    df["cluster_kmeans"] = kmeans.fit_predict(df[features])

    gmm = GaussianMixture(n_components=4, n_init=10, random_state=1234)
    df["cluster_gmm"] = gmm.fit_predict(df[features])

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    df["cluster_dbscan"] = dbscan.fit_predict(df[features])

    print(f"Silhouette Score for k-Means: {silhouette_score(df[features], df['cluster_kmeans'])}")
    print(f"Silhouette Score for GMM: {silhouette_score(df[features], df['cluster_gmm'])}")
    print(f"Silhouette Score for DBSCAN: {silhouette_score(df[features], df['cluster_dbscan'])}")

    return df


def linear_regression(df, features):
    """Perform linear regression and evaluate the model."""
    X = df[features]
    y = df["Like.n"]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    model_train = sm.OLS(y_train, X_train).fit()
    y_pred = model_train.predict(X_test)
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"R-squared: {r2_score(y_test, y_pred)}")


def visualize_data(df, features, df_pca):
    """Visualize data using heatmaps and scatter plots."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=df["cluster_kmeans"], palette='viridis')
    plt.title('k-Means Clustering with PCA')
    plt.show()


def hyperparameter_tuning(df, features):
    """Perform hyperparameter tuning for k-Means and GMM."""
    param_grid_kmeans = {'n_clusters': [2, 3, 4, 5, 6]}
    grid_kmeans = GridSearchCV(KMeans(n_init=10, random_state=1234), param_grid_kmeans, cv=5)
    grid_kmeans.fit(df[features])
    print(f"Best Params for k-Means: {grid_kmeans.best_params_}")

    param_grid_gmm = {'n_components': [2, 3, 4, 5, 6]}
    grid_gmm = GridSearchCV(GaussianMixture(n_init=10, random_state=1234), param_grid_gmm, cv=5)
    grid_gmm.fit(df[features])
    print(f"Best Params for GMM: {grid_gmm.best_params_}")


def main():
    file_path = "mcdonalds.csv"
    df = load_data(file_path)
    if df is not None:
        df, features = preprocess_data(df)
        df_pca = apply_pca(df, features)
        df = perform_clustering(df, features)
        linear_regression(df, features)
        visualize_data(df, features, df_pca)
        hyperparameter_tuning(df, features)


if __name__ == "__main__":
    main()
