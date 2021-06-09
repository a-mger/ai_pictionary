import io
from tensorflow.python.lib.io import file_io
import numpy as np
from tensorflow import keras
#from ai_pictionary import xpred
import xpred
import pandas as pd

def get_test_data():
    """method to get the test data (or a portion of it) from google cloud bucket
    To predict we can either obtain predictions from train data or from test data"""
    f = io.BytesIO(
            file_io.read_file_to_string(
                f'gs://ai_pictionary_bucket/data/X_test',
                binary_mode=True))
    X_test = np.load(f)
    print("loaded X_test")
    g = io.BytesIO(
        file_io.read_file_to_string(f'gs://ai_pictionary_bucket/data/y_test',
                                    binary_mode=True))
    y_test = np.load(g)
    print("loaded y_test")

    return X_test, y_test


def get_labels():
    """method to get the test data (or a portion of it) from google cloud bucket
    To predict we can either obtain predictions from train data or from test data"""
    f = io.BytesIO(
        file_io.read_file_to_string(f'gs://ai_pictionary_bucket/data/labels',
                                    binary_mode=True))
    labels = np.load(f)
    print("loaded labels")

    return labels


def get_model():
    pipeline = keras.models.load_model("gs://ai_pictionary_bucket/models/model.joblib")
    return pipeline


def predict(pipeline, X_pred):
    y_pred = pipeline.predict(X_pred)
    return y_pred



if __name__ == '__main__':
    #X_test, y_test = get_test_data()
    model = get_model()
    labels = get_labels()
    X_pred = xpred.X_pred
    X_pred = X_pred.reshape((1,28,28,1))
    y_pred = model.predict(X_pred)
    y_pred = pd.DataFrame(y_pred)
    maxValueIndex = y_pred.idxmax(axis=1)
    print(labels[maxValueIndex[0]])
    #print(X_test.shape)
    #print(y_test.shape)

    #model.evaluate(X_test, y_test)
