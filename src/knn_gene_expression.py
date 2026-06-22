"""KNN classifier on gene expression data.

Reproduces the workflow from KNN_Gene_Expression.ipynb as a plain script:
load the data, scale the features, fit KNN, run an elbow sweep over K, and
finally let a cross validated grid search pick K inside a pipeline.
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "gene_expression.csv")


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    X = df.drop("Cancer Present", axis=1)
    y = df["Cancer Present"]
    return X, y


def elbow_sweep(scaled_X_train, y_train, scaled_X_test, y_test, k_range=range(1, 30)):
    """Return the test error rate for each value of K."""
    error_rates = []
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(scaled_X_train, y_train)
        preds = model.predict(scaled_X_test)
        error_rates.append(1 - accuracy_score(y_test, preds))
    return error_rates


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=101
    )

    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)

    # Single model with K = 9, matching the notebook's first fit.
    knn = KNeighborsClassifier(n_neighbors=9)
    knn.fit(scaled_X_train, y_train)
    y_pred = knn.predict(scaled_X_test)
    print("Confusion matrix (K=9):")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Elbow sweep over K.
    errors = elbow_sweep(scaled_X_train, y_train, scaled_X_test, y_test)
    best_k = list(range(1, 30))[errors.index(min(errors))]
    print(f"Lowest test error in the sweep at K = {best_k}")

    # Grid search over a scaler plus KNN pipeline, selecting K by cross validation.
    pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
    param_grid = {"knn__n_neighbors": list(range(1, 20))}
    search = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy")
    search.fit(X_train, y_train)
    chosen_k = search.best_estimator_.named_steps["knn"].n_neighbors
    print(f"Grid search selected K = {chosen_k}")
    print(classification_report(y_test, search.predict(X_test)))


if __name__ == "__main__":
    main()
