from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow import keras
import pandas as pd
import json
import random
from tensorflow.python.lib.io import file_io
import io

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


@app.get("/getpic")
def predict(label):
    f = io.BytesIO(
                file_io.read_file_to_string(
                    f'gs://quickdraw_dataset/full/numpy_bitmap/' + label +".npy",
                    binary_mode=True))
    sample_array = np.load(f)
    random_number = random.randint(0,len(sample_array))
    random_pic = sample_array[random_number,:].reshape(28,28)
    random_pic_list = random_pic.tolist()
    random_pic_json = json.dumps(random_pic_list)
    return random_pic_json
