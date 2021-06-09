from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import io
from tensorflow.python.lib.io import file_io
import numpy as np
from tensorflow import keras
import pandas as pd

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
def predict(X_pred):

    X_pred = np.array(X_pred)

    g = io.BytesIO(
        file_io.read_file_to_string(f'gs://ai_pictionary_bucket/data/labels',
                                    binary_mode=True))
    labels = np.load(g)
    model = keras.models.load_model(
        "gs://ai_pictionary_bucket/models/model.joblib")
    y_pred = model.predict(X_pred)
    y_pred = pd.DataFrame(y_pred)
    maxValueIndex = y_pred.idxmax(axis=1)
    print(labels[maxValueIndex[0]])
    y_pred = model.predict(X_pred)
    return y_pred, labels