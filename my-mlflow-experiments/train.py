import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraud_detection_model")

df = pd.read_csv("data/clean.csv")

X = df[["amount"]]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

with mlflow.start_run(run_name = "Logistic Regression, C = 0.1"):
    model = LogisticRegression(C=0.1)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    
    mlflow.log_params({"model": "LogisticRegression", "C": 0.1})
    mlflow.log_metrics({"auc_roc": auc_roc, "auc_pr": auc_pr})
    mlflow.sklearn.log_model(model, "model", registered_model_name="my_fraud_detector")
    
    print(f"LR C=0.1 | AUC-ROC: {auc_roc:.4f} | AUC-PR: {auc_pr:.4f}")

with mlflow.start_run(run_name = "Logistic Regression, C = 10.0"):
    model = LogisticRegression(C=10.0)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    
    mlflow.log_params({"model": "LogisticRegression", "C": 10.0})
    mlflow.log_metrics({"auc_roc": auc_roc, "auc_pr": auc_pr})
    mlflow.sklearn.log_model(model, "model", registered_model_name="my_fraud_detector")
    
    print(f"LR C=10.0 | AUC-ROC: {auc_roc:.4f} | AUC-PR: {auc_pr:.4f}")

    
with mlflow.start_run(run_name = "Random Forest Classifier"):
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    
    mlflow.log_params({"model": "RandomForest", "n_estimators": 50})
    mlflow.log_metrics({"auc_roc": auc_roc, "auc_pr": auc_pr})
    mlflow.sklearn.log_model(model, "model", registered_model_name="my_fraud_detector")
    
    print(f"RF n=50 | AUC-ROC: {auc_roc:.4f} | AUC-PR: {auc_pr:.4f}")


from mlflow.tracking import MlflowClient

def select_best_model():
    client = MlflowClient()
    
    runs = mlflow.search_runs(experiment_names=["fraud_detection_model"])
    best_run = runs.sort_values("metrics.auc_pr", ascending=False).iloc[0]
    
    print(f"Best model: {best_run['tags.mlflow.runName']}")
    print(f"AUC-PR: {best_run['metrics.auc_pr']:.4f}")
    print(f"AUC-ROC: {best_run['metrics.auc_roc']:.4f}")
    
    best_version = client.search_model_versions(f"name='my_fraud_detector'")
    best_version = [v for v in best_version if v.run_id == best_run["run_id"]][0]
    
    client.transition_model_version_stage(
        name="my_fraud_detector",
        version=best_version.version,
        stage="Staging"
    )
    
    print(f"Model version {best_version.version} transitioned to Staging")

select_best_model()