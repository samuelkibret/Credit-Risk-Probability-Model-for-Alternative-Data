import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy import sparse
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn

def load_data():
    X = sparse.load_npz("data/processed/X_transformed.npz")
    y = np.load("data/processed/y_transformed.npy", allow_pickle=True)
    return X, y

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None
    }
    return metrics

def train_and_track_models():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42)
    }

    param_grids = {
        "LogisticRegression": {"C": [0.01, 0.1, 1, 10]},
        "RandomForest": {"n_estimators": [50, 100], "max_depth": [5, 10, None]}
    }

    best_model = None
    best_score = 0

    mlflow.set_experiment("credit_risk_modeling")

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        grid_search = GridSearchCV(model, param_grids[model_name], cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_estimator = grid_search.best_estimator_
        metrics = evaluate_model(best_estimator, X_test, y_test)

        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_estimator, artifact_path="model")

        print(f"{model_name} metrics: {metrics}")

        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model = best_estimator

    print(f"Best model: {best_model} with ROC-AUC: {best_score}")

if __name__ == "__main__":
    train_and_track_models()
