from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from google.cloud import storage
from ai_pictionary.params import MODEL_NAME

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

    bucket = "ai_pictionary_bucket"

    client = storage.Client().bucket(bucket)
    storage_location = 'models/{}/v1/{}'.format(MODEL_NAME, 'model.joblib')
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    pipeline = joblib.load('model.joblib')
    y_pred = pipeline.predict(X_pred)
    return y_pred