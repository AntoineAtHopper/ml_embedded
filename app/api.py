'''
## Load Balancing 
$num_workers = (2 x $num_cores) + 1
gunicorn main:app -w $num_workers -k uvicorn.workers.UvicornWorker -b "0.0.0.0:8888"

gunicorn main:app -w 2 -k uvicorn.workers.UvicornWorker -b "0.0.0.0:8888"

[Doc](https://fastapi.tiangolo.com/deployment/concepts/)


## Background Task (For training)
[Doc](https://fastapi.tiangolo.com/tutorial/background-tasks/)


## Docker
[Doc](https://fastapi.tiangolo.com/deployment/docker/)
'''

# Imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi_utils.tasks import repeat_every
from dotenv import load_dotenv, find_dotenv
from filelock import Timeout, FileLock
from transformers import pipeline
import pandas as pd
import requests
import uvicorn
import json
import time
import os
import re

from PIL import Image

from app.tasks import save_query, training_pipeline
from app.model import Model

#load_dotenv(find_dotenv())
#prefix = os.getenv("CLUSTER_ROUTE_PREFIX", "")
df = None
CSV_QUERIES = "queries.csv"
CONFIG_FILE = "config.json"


main_worker = None
worker_id = os.getpid()
# Instantiate Model
model = Model()

# Instantiate FastAPI app
app = FastAPI()

""" Needed for Cross-Origin requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
)
"""

# Main
@app.get("/")
def main():
    return "Welcome on ML Embedded API!"


# Predict
@app.post("/predict")
def predict(background_tasks: BackgroundTasks, image_url: str = ""):
    image = Image.open(requests.get(image_url, stream=True).raw)
    prediction = model.predict(image)
    background_tasks.add_task(save_query, image_url, prediction)
    return {"results": prediction}


# Train the model every 24 hours.
@app.on_event("startup")
@repeat_every(seconds=60*60*24)
def train():
    global main_worker
    # Get or Set Main Worker
    if main_worker is None:
        lock = FileLock(CONFIG_FILE + ".lock")
        with lock:
            with open(CONFIG_FILE, "r+") as f:
                config = json.load(f)
                print("...")
                if config["MAIN_WORKER"] == None:
                    config["MAIN_WORKER"] = worker_id
                    f.seek(0, os.SEEK_SET)
                    json.dump(config, f)
                main_worker = config["MAIN_WORKER"]
        print("[*] main_worker:", main_worker)
    if worker_id == main_worker:
        print("[*] Training")
        training_pipeline()


@app.on_event("shutdown")
def shutdown():
    # Reset Main Worker
    if main_worker == worker_id:
        lock = FileLock(CONFIG_FILE + ".lock")
        with lock:
            with open(CONFIG_FILE, "w+") as f:
                config = json.load(f)
                config["MAIN_WORKER"] = None
                json.dump(config, f)