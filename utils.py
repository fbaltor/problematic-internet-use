import json
import os
import pickle
import shutil
from typing import Dict

import kagglehub
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from kagglehub.config import set_kaggle_credentials
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from constants import DATASET_PATH, ID_COL, LABEL_COL


def download_data() -> None:
    # Fetch Credentials
    credentials = json.load(
        open("credentials.json")
    )  # Format {"username":"...", "api_key":"..."}
    set_kaggle_credentials(**credentials)

    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATASET_PATH):
        path = kagglehub.competition_download(
            "child-mind-institute-problematic-internet-use"
        )
        shutil.move(path, DATASET_PATH)


def get_cols_from_df(df: pd.DataFrame, features) -> pd.DataFrame:
    return df[[ID_COL] + features + [LABEL_COL]].copy(deep=True)


def get_experiment_folder(experiment_name: str) -> str:
    return f"experiments/{experiment_name.replace(' ', '_')}"


def load_user_events_df(user_id: str, is_train: bool = True) -> pd.DataFrame:
    mode = "train" if is_train else "test"
    path = f"{DATASET_PATH}series_{mode}.parquet/id={user_id}"
    events_df = pd.DataFrame()
    if not os.path.exists(path):
        return events_df

    files = os.listdir(path)
    for file in files:
        if file.endswith(".parquet"):
            events_df = pd.concat([events_df, pd.read_parquet(f"{path}/{file}")])
    return events_df


def save_image(experiment_name: str, filename: str) -> None:
    experiment_folder = get_experiment_folder(experiment_name)
    os.makedirs(experiment_folder, exist_ok=True)
    plt.savefig(f"{experiment_folder}/{filename}.png")


def save_text(experiment_name: str, filename: str, text: str) -> None:
    experiment_folder = get_experiment_folder(experiment_name)
    os.makedirs(experiment_folder, exist_ok=True)
    with open(f"{experiment_folder}/{filename}.txt", "w") as f:
        f.write(text)


def save_model(model, experiment_name: str):
    experiment_folder = get_experiment_folder(experiment_name)
    os.makedirs(experiment_folder, exist_ok=True)
    with open(f"{experiment_folder}/model.pkl", "wb") as f:
        pickle.dump(model, f)


def save_metadata(experiment_name: str, filename: str, metadata: Dict):
    experiment_folder = get_experiment_folder(experiment_name)
    os.makedirs(experiment_folder, exist_ok=True)
    with open(f"{experiment_folder}/{filename}.json", "w") as f:
        json.dump(metadata, f, indent=4)


def compute_metrics(y_true, y_pred):
    pred_labels = [np.argmax(x) for x in y_pred]
    return {
        "auc_score": roc_auc_score(y_true, y_pred, multi_class="ovr"),
        "accuracy": accuracy_score(y_true, pred_labels),
    }


def save_results_locally(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    experiment_name: str,
    experiment_metadata: Dict,
):
    # Save Training Metrics
    plt.figure()
    lgb.plot_metric(model)
    save_image(experiment_name, "training_metrics")
    # Save Model
    save_model(model, experiment_name)
    # Save Metadata
    save_metadata(
        experiment_name=experiment_name,
        filename="metadata",
        metadata=experiment_metadata,
    )
    # Save Predictions
    train_predictions = model.predict_proba(X_train)
    save_text(experiment_name, "train_predictions", str(train_predictions.tolist()))

    test_predictions = model.predict_proba(X_test)
    save_text(experiment_name, "test_predictions", str(test_predictions.tolist()))
    # Save Feature Importance Gain
    plt.figure()
    lgb.plot_importance(
        model, importance_type="gain", max_num_features=10, figsize=(10, 5)
    )
    save_image(experiment_name, "gain_feature_importance")
    # Save Feature Importance Split
    plt.figure()
    lgb.plot_importance(
        model,
        importance_type="split",
        figsize=(7, 6),
        title="LightGBM Feature Importance (Split)",
    )
    save_image(experiment_name, "split_feature_importance")
    # Save Evaluation Metrics
    plt.figure()
    train_metrics = compute_metrics(y_train, train_predictions)
    test_metrics = compute_metrics(y_test, test_predictions)
    metrics = {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    save_metadata(
        experiment_name=experiment_name, filename="eval_metrics", metadata=metrics
    )
    # Save Confusion Matrix
    plt.figure()
    test_predictions_labels = [np.argmax(x) for x in test_predictions]
    matrix = confusion_matrix(y_test, test_predictions_labels)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    save_image(experiment_name, "confusion_matrix")
    # Save Train Predictions Histograms
    plt.figure()
    plt.title("Train Predictions Histogram")
    sns.histplot(train_predictions, bins=100)
    save_image(experiment_name, "train_predictions_histogram")
    # Save Test Predictions Histograms
    plt.figure()
    plt.title("Test Predictions Histogram")
    sns.histplot(test_predictions, bins=100)
    save_image(experiment_name, "test_predictions_histogram")
    print(train_metrics)
    print(test_metrics)


def save_to_mlflow(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    experiment_name: str,
    experiment_metadata: Dict,
):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        # Save Training Metrics
        fig, ax = plt.subplots()
        lgb.plot_metric(model, ax=ax)
        mlflow.log_figure(fig, "training.png")
        # Save Model
        mlflow.sklearn.log_model(model, "model")
        # Save Evaluation Metrics
        train_predictions = model.predict_proba(X_train)
        test_predictions = model.predict_proba(X_test)
        train_metrics = compute_metrics(y_train, train_predictions)
        test_metrics = compute_metrics(y_test, test_predictions)
        metrics = {}
        for key, value in train_metrics.items():
            metrics[f"train_{key}"] = round(value, 4)
        for key, value in test_metrics.items():
            metrics[f"test_{key}"] = round(value, 4)
        mlflow.log_metrics(metrics)
        # Save Predictions
        train_predictions = model.predict_proba(X_train)
        test_predictions = model.predict_proba(X_test)
        predictions = {
            "train_predictions": str(train_predictions.tolist()),
            "test_predictions": str(test_predictions.tolist()),
        }
        mlflow.log_dict(predictions, "predictions.json")
        # Save Feature Importance Gain
        fig, ax = plt.subplots()
        lgb.plot_importance(
            model, ax=ax, importance_type="gain", max_num_features=10, figsize=(10, 5)
        )
        mlflow.log_figure(fig, "feature_importance__gain.png")
        # Save Feature Importance Slip
        fig, ax = plt.subplots()
        lgb.plot_importance(
            model, ax=ax, importance_type="split", max_num_features=10, figsize=(10, 5)
        )
        mlflow.log_figure(fig, "feature_importance__split.png")
        # Save Confusion Matrix
        fig, ax = plt.subplots()
        test_predictions_labels = [np.argmax(x) for x in test_predictions]
        matrix = confusion_matrix(y_test, test_predictions_labels)
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        mlflow.log_figure(fig, "confusion_matrix.png")
        # Save Train Predictions Histograms
        fig, ax = plt.subplots()
        sns.histplot(train_predictions, bins=100)
        mlflow.log_figure(fig, "train_predictions_histogram.png")
        # Save Test Predictions Histograms
        fig, ax = plt.subplots()
        sns.histplot(test_predictions, bins=100)
        mlflow.log_figure(fig, "test_predictions_histogram.png")
        # Save Parameters
        mlflow.log_params(model.get_params())
        # Save Model Metadata
        mlflow.log_dict(experiment_metadata, "metadata.json")


def save_results(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    experiment_name: str,
    experiment_metadata: Dict,
    save_on_mlflow: bool = True,
    save_locally: bool = False,
):
    if save_locally:
        os.makedirs("experiments", exist_ok=True)
        assert not os.path.exists(
            get_experiment_folder(experiment_name)
        ), "Experiment already exists locally"
        save_results_locally(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            experiment_name=experiment_name,
            experiment_metadata=experiment_metadata,
        )
    if save_on_mlflow:
        save_to_mlflow(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            experiment_name=experiment_name,
            experiment_metadata=experiment_metadata,
        )
