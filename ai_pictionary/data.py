#import params
from ai_pictionary import params
import io
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.backend import expand_dims
import os
from tensorflow.python.lib.io import file_io
from random import sample
from ai_pictionary import categories
from tensorflow.keras.backend import expand_dims


def get_data(max_items_per_class=5000, max_labels=10):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    #making list of filenames to load from gcl

    filenames = sample(categories.labels_150, max_labels)

    #creating X, y and index to use for the loop and a label_names for making a list to check the y-label
    X = np.empty([0, 784])
    y = np.empty([0])
    index = 0

    for file in filenames:
        f = io.BytesIO(
            file_io.read_file_to_string(
                f'gs://quickdraw_dataset/full/numpy_bitmap/' + file +".npy",
                binary_mode=True))
        data = np.load(f)
        data = data[0:max_items_per_class, :]
        label = np.full(data.shape[0], index)
        y = np.append(y, label)
        X = np.concatenate((X, data), axis=0)
        print(index)
        index += 1
    #reshaping to (28,28)
    X = X.reshape(len(X),28,28)
    #making test and training set with train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
    X_train = expand_dims(X_train, axis=-1)
    X_test = expand_dims(X_test, axis=-1)
    print("data fetched")
    np.save(
        file_io.FileIO(
            f'gs://{params.BUCKET_NAME}/{params.BUCKET_TRAIN_DATA_PATH}/labels.npy',
            'w'), filenames)
    # np.save(
    #     file_io.FileIO(
    #         f'gs://{params.BUCKET_NAME}/{params.BUCKET_TRAIN_DATA_PATH}/X_train.npy',
    #         'w'), X_train)
    # np.save(
    #     file_io.FileIO(
    #         f'gs://{params.BUCKET_NAME}/{params.BUCKET_TRAIN_DATA_PATH}/X_test.npy',
    #         'w'), X_test)
    # np.save(
    #     file_io.FileIO(
    #         f'gs://{params.BUCKET_NAME}/{params.BUCKET_TRAIN_DATA_PATH}/y_train.npy',
    #         'w'), y_train)
    # np.save(
    #     file_io.FileIO(
    #         f'gs://{params.BUCKET_NAME}/{params.BUCKET_TRAIN_DATA_PATH}/y_test.npy',
    #         'w'), y_test)

    print("saved labels")

    return X_train, X_test, y_train, y_test, filenames

def get_data_KNN(max_items_per_class=5000, max_labels=10):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    #making list of filenames to load from gcl

    filenames = sample(categories.labels, max_labels)

    #creating X, y and index to use for the loop and a label_names for making a list to check the y-label
    X = np.empty([0, 784])
    y = np.empty([0])
    index = 0

    for file in filenames:
        f = io.BytesIO(
            file_io.read_file_to_string(
                f'gs://quickdraw_dataset/full/numpy_bitmap/' + file +".npy",
                binary_mode=True))
        data = np.load(f)
        data = data[0:max_items_per_class, :]
        label = np.full(data.shape[0], index)
        y = np.append(y, label)
        X = np.concatenate((X, data), axis=0)
        print(index)
        index += 1

    return X, y, filenames


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, labels = get_data()
    print(len(X_train))
