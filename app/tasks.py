import pandas as pd
import os
import time
import requests
import json
from app.model import Model

from filelock import Timeout, FileLock

# Load configuration
def load_config():
    lock = FileLock(CONFIG_FILE + ".lock")
    with lock:
        with open(CONFIG_FILE) as f:
            config = json.load(f)
    return config


# Load configuration variables
CSV_QUERIES = "queries.csv"
CSV_TRAINING = "training.csv"
CONFIG_FILE = "config.json"

model = Model()
config = load_config()
df = pd.read_csv(CSV_QUERIES) if os.path.exists(CSV_QUERIES) else pd.DataFrame(columns=["image_url", "prediction", "time"])
df_training = pd.read_csv(CSV_TRAINING) if os.path.exists(CSV_TRAINING) else pd.DataFrame(columns=["label", "image_url"])


# Save query
def save_query(image_url, prediction):
    global df
    # Add query
    df = df.append({"image_url": image_url, "prediction": prediction, "time": time.time()}, ignore_index=True)
    # Save df
    df.to_csv(CSV_QUERIES, index=False)


def fetch_images(query):
    # https://unsplash.com/documentation
    params = {
        'query': query,
        'page': 1,
        'per_page': 10
    }
    headers = {
        'Authorization': "Client-ID irUGrRYmVrNx44BsgAzzAVAd4NU-S2v9i3o1EiQfNGE"
    }
    req = requests.get("https://api.unsplash.com/search/photos", params=params, headers=headers)
    res = json.loads(req.text)
    images_url = [r["links"]["download"] for r in res["results"]]
    return images_url


def training_pipeline(k=10):
    print("training_pipeline")
    global df_training
    # Get a set of predicted class
    predicted_class = df[df["time"] > config["LAST_UPDATE"]]["prediction"].drop_duplicates().sort_values(ascending=False)[:k]
    print("predicted_class", predicted_class)
    # Fetch images url
    images_url = [(p_class, url) for p_class in predicted_class for url in fetch_images(p_class) if url not in df_training["image_url"]]
    df_training_ = pd.DataFrame(images_url, columns=df_training.columns)
    # Train the model
    model.train(df_training_)
    # Update df_training
    df_training = df_training.append(df_training_, ignore_index=True)
    df_training.to_csv(CSV_TRAINING, index=False)
    print("df_train", df_training)
    # Update config
    config["LAST_UPDATE"] = time.time()
    with open(CONFIG_FILE, "w+") as f:
        json.dump(config, f)

    