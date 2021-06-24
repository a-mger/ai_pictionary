from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow import keras
import pandas as pd
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def index():
    return {"greeting": "Hello world"}


@app.get("/predict")
def predict(img_frontend, model_type):
    """gets image data as string and load it into a numpy array
    load model depening on input (CNN50 / CNN150 / CNN250)
    predict and get the top 5 probabilities"""
    img_json = json.loads(img_frontend)
    X_pred = np.array(img_json).reshape((1, 28, 28, 1))
    X_pred = X_pred/255
    labels = np.load(f"model/{model_type}/labels.npy")
    model = keras.models.load_model(f"model/{model_type}")
    y_pred = model.predict(X_pred)
    y_pred = pd.DataFrame(y_pred)
    y_pred = y_pred.sort_values(by=0, axis=1, ascending=False)
    top5 = y_pred.iloc[:, 0:5]
    for index in top5.columns:
        top5 = top5.rename(columns={index: labels[index]})
    top5_json = top5.to_json()
    return top5_json
