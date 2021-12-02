'''
## Load Balancing 
$num_workers = (2 x $num_cores) + 1
gunicorn main:app -w $num_workers -k uvicorn.workers.UvicornWorker -b "0.0.0.0:8888"
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
    print("[*] Training")
    training_pipeline()