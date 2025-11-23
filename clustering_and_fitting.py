import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit


def plot_relational_plot(df):
    """Relational plot: Sales Amount vs Product_ID"""
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Product_ID", y="Sales_Amount")
    plt.title("Relational Plot: Sales Amount Across Products")
    plt.xlabel("Product_ID")
    plt.ylabel("Sales_Amount")
    plt.savefig("relational_plot.png")
    plt.show()


def plot_categorical_plot(df):
    """Categorical plot: Average Sales Amount by Product Category"""
    fig, ax = plt.subplots()
    cat = df.groupby("Product_Category")["Sales_Amount"].mean().reset_index()
    sns.barplot(data=cat, x="Product_Category", y="Sales_Amount", palette="Set2")
    plt.title("Average Sales Amount by Category")
    plt.xlabel("Product_Category")
    plt.ylabel("Average Sales_Amount")
    plt.savefig("categorical_plot.png")
    plt.show()


def plot_statistical_plot(df):
    """Statistical plot: Correlation heatmap of numeric columns"""
    fig, ax = plt.subplots()
    num = df[["Sales_Amount", "Quantity_Sold", "Unit_Cost", "Unit_Price", "Discount"]]
    sns.heatmap(num.corr(), annot=True, cmap="Blues")
    plt.title("Correlation Heatmap")
    plt.savefig("statistical_plot.png")
    plt.show()


def statistical_analysis(df, col: str):
    """Computes 4 statistical moments"""
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col])
    excess_kurtosis = ss.kurtosis(df[col])
    return mean, stddev, skew, excess_kurtosis


def writing(moments, col):
    """Prints statistical moments with dynamic interpretation"""
    print(f"\nFor the attribute {col}:")
    print(f"Mean = {moments[0]:.2f}, Std Dev = {moments[1]:.2f}, "
          f"Skew = {moments[2]:.2f}, Excess Kurtosis = {moments[3]:.2f}")
    skew_str = "not skewed" if abs(moments[2]) < 0.5 else ("right skewed" if moments[2] > 0 else "left skewed")
    kurt_str = "mesokurtic" if abs(moments[3]) < 1 else ("leptokurtic" if moments[3] > 1 else "platykurtic")
    print(f"The data was {skew_str} and {kurt_str}.")


def preprocessing(df):
    """Preprocess dataset"""
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates()
    df = df.fillna(df.mean(numeric_only=True))
    print("\nDataset Description:\n", df.describe())
    print("\nCorrelation:\n", df.corr(numeric_only=True))
    return df


def perform_clustering(df, col1, col2):
    """Performs KMeans clustering on two numeric columns"""
    data = df[[col1, col2]].values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    # Elbow plot
    inertias = []
    ks = range(1, 6)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=0)
        km.fit(scaled)
        inertias.append(km.inertia_)
    fig, ax = plt.subplots()
    sns.lineplot(x=list(ks), y=inertias, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.savefig("elbow_plot.png")
    plt.show()

    km = KMeans(n_clusters=3, random_state=0)
    labels = km.fit_predict(scaled)
    centers = km.cluster_centers_
    xkmeans = centers[:, 0]
    ykmeans = centers[:, 1]

    return labels, scaled, xkmeans, ykmeans, centers


def plot_clustered_data(labels, data, xkmeans, ykmeans, centres):
    """Plot clustered data with cluster centers"""
    fig, ax = plt.subplots()
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap="viridis")
    plt.scatter(xkmeans, ykmeans, c="red", s=100, marker="X")
    plt.title("KMeans Clustering")
    plt.xlabel("Unit_Price")
    plt.ylabel("Unit_Cost")
    plt.savefig("clustering.png")
    plt.show()


def perform_fitting(df, col1, col2):
    """Linear curve fitting using scipy's curve_fit"""
    x = df[col1].values
    y = df[col2].values

    def linear_model(x, a, b):
        return a * x + b

    popt, _ = curve_fit(linear_model, x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = linear_model(x_line, *popt)

    return (x, y), x_line, y_line


def plot_fitted_data(data, x, y):
    """Plot original data and fitted line"""
    x_data, y_data = data
    fig, ax = plt.subplots()
    plt.scatter(x_data, y_data, label="Actual", color="blue")
    plt.plot(x, y, label="Fitted", color="red")
    plt.xlabel("Unit_Price")
    plt.ylabel("Sales_Amount")
    plt.title("Fitted Data: Unit_Price vs Sales_Amount")
    plt.legend()
    plt.savefig("fitting.png")
    plt.show()


def main():
    df = pd.read_csv("data.csv")
    df = preprocessing(df)

    col = "Sales_Amount"

    plot_relational_plot(df)
    plot_categorical_plot(df)
    plot_statistical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)

    clustering_results = perform_clustering(df, "Unit_Price", "Unit_Cost")
    plot_clustered_data(*clustering_results)

    fitting_results = perform_fitting(df, "Unit_Price", "Sales_Amount")
    plot_fitted_data(*fitting_results)

    print("\nAssignment Completed Successfully!")


if __name__ == "__main__":
    main()
