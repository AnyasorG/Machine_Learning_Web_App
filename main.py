import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit configuration
st.set_page_config(page_title="Machine Learning Classifier Comparison", layout="wide")

# Title and description
st.title("🔍 Machine Learning Classifier Comparison App")
st.write(
    """
This app allows you to explore different classifiers on popular datasets.
You can adjust classifier parameters and see how they affect performance.
"""
)

# Sidebar for dataset selection
st.sidebar.header("Dataset and Classifier Selection")

def main():
    # Dataset selection
    dataset_name = st.sidebar.selectbox(
        "Select Dataset", ("Iris", "Breast Cancer", "Wine")
    )

    # Classifier selection
    classifier_name = st.sidebar.selectbox(
        "Select Classifier", ("K-Nearest Neighbors", "Support Vector Machine", "Random Forest")
    )

    # Load dataset
    X, y = load_dataset(dataset_name)
    st.write(f"## {dataset_name} Dataset")
    st.write("Shape of dataset:", X.shape)
    st.write("Number of classes:", len(np.unique(y)))

    # Show dataset
    if st.checkbox("Show raw data"):
        st.write(pd.DataFrame(X).head())

    # Feature Scaling
    if st.sidebar.checkbox("Scale Features"):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Get classifier parameters
    params = get_classifier_params(classifier_name)

    # Initialize classifier
    clf = get_classifier(classifier_name, params)

    # Model training and evaluation
    accuracy, report = evaluate_model(clf, X, y)

    # Display results
    st.write(f"### Classifier: {classifier_name}")
    st.write(f"Accuracy: **{accuracy:.2f}%**")
    st.write("Classification Report:")
    st.text(report)

    # Plot PCA
    plot_pca(X, y)

def load_dataset(name):
    if name == "Iris":
        data = datasets.load_iris()
    elif name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    return data.data, data.target

def get_classifier_params(name):
    params = dict()
    if name == "K-Nearest Neighbors":
        K = st.sidebar.slider("Number of Neighbors (K)", 1, 15, value=5)
        params["n_neighbors"] = K
    elif name == "Support Vector Machine":
        C = st.sidebar.slider("Regularization Parameter (C)", 0.01, 10.0, value=1.0)
        kernel = st.sidebar.selectbox("Kernel", ("linear", "rbf", "poly"))
        params["C"] = C
        params["kernel"] = kernel
    elif name == "Random Forest":
        n_estimators = st.sidebar.slider("Number of Trees", 10, 100, step=10, value=100)
        max_depth = st.sidebar.slider("Max Depth", 1, 20, value=10)
        params["n_estimators"] = n_estimators
        params["max_depth"] = max_depth
    return params

def get_classifier(name, params):
    if name == "K-Nearest Neighbors":
        clf = KNeighborsClassifier(**params)
    elif name == "Support Vector Machine":
        clf = SVC(**params)
    elif name == "Random Forest":
        clf = RandomForestClassifier(**params, random_state=42)
    return clf

def evaluate_model(clf, X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Train model
    clf.fit(X_train, y_train)
    # Predict
    y_pred = clf.predict(X_test)
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    # Confusion matrix
    st.write("Confusion Matrix:")
    plot_confusion(y_test, y_pred)
    return accuracy, report

def plot_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    st.pyplot(fig)

def plot_pca(X, y):
    st.write("### Data Visualization")
    pca = PCA(2)
    X_projected = pca.fit_transform(X)
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]
    fig, ax = plt.subplots()
    scatter = ax.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
    legend = ax.legend(
        *scatter.legend_elements(), title="Classes", loc="upper right"
    )
    ax.add_artist(legend)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
