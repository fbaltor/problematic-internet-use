import json
import os
import pickle
import shutil
from typing import Dict

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kagglehub.config import set_kaggle_credentials
from sklearn.metrics import accuracy_score, roc_auc_score

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
    experiment_name = experiment_name.replace(" ", "_")
    os.makedirs(f"experiments/{experiment_name}", exist_ok=True)
    plt.savefig(f"experiments/{experiment_name}/{filename}.png")


def save_text(experiment_name: str, filename: str, text: str) -> None:
    experiment_name = experiment_name.replace(" ", "_")
    with open(f"experiments/{experiment_name}/{filename}.txt", "w") as f:
        f.write(text)


def save_model(model, experiment_name: str):
    experiment_name = experiment_name.replace(" ", "_")
    os.makedirs(f"experiments/{experiment_name}", exist_ok=True)
    with open(f"experiments/{experiment_name}/model.pkl", "wb") as f:
        pickle.dump(model, f)


def save_metadata(experiment_name: str, filename: str, metadata: Dict):
    experiment_name = experiment_name.replace(" ", "_")
    with open(f"experiments/{experiment_name}/{filename}.json", "w") as f:
        json.dump(metadata, f, indent=4)


def compute_metrics(y_true, y_pred):
    pred_labels = [np.argmax(x) for x in y_pred]
    return {
        "auc_score": roc_auc_score(y_true, y_pred, multi_class="ovr"),
        "accuracy": accuracy_score(y_true, pred_labels),
    }
