#import params
from ai_pictionary import params
import io
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.backend import expand_dims
import os
from tensorflow.python.lib.io import file_io




def get_data(max_items_per_class=5000):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    #making list of filenames to load from gcl
    filenames = [
        "full_numpy_bitmap_alarm clock.npy", "full_numpy_bitmap_bicycle.npy",
        "full_numpy_bitmap_boomerang.npy", "full_numpy_bitmap_bread.npy",
        "full_numpy_bitmap_broccoli.npy", "full_numpy_bitmap_lantern.npy",
        "full_numpy_bitmap_lightning.npy", "full_numpy_bitmap_streetlight.npy",
        "full_numpy_bitmap_tennis racquet.npy", "full_numpy_bitmap_tractor.npy"
    ]

    #creating X, y and index to use for the loop and a label_names for making a list to check the y-label
    X = np.empty([0, 784])
    y = np.empty([0])
    index = 0
    label_names = []

    for fname in filenames:
        label_name = fname
        label_name = os.path.splitext(label_name)[0]
        label_name = label_name.replace("full_numpy_bitmap_", "")
        label_names.append(
            label_name) if label_name not in label_names else label_names
        # f = io.BytesIO(
        #     file_io.read_file_to_string(
        #         f'gs://{params.BUCKET_NAME}/{params.BUCKET_TRAIN_DATA_PATH}/' +
        #         fname,
        #         binary_mode=True))
        f = io.BytesIO(
            file_io.read_file_to_string(
                f'gs://quickdraw_dataset/full/numpy_bitmap/' + label_name+".npy",
                binary_mode=True))
        data = np.load(f)
        data = data[0:max_items_per_class, :]
        label = np.full(data.shape[0], index)
        y = np.append(y, label)
        X = np.concatenate((X, data), axis=0)
        index += 1
    "quickdraw_dataset/full/numpy_bitmap"
    #reshaping to (28,28)
    X = X.reshape(len(X),28,28)
    np.save(
        file_io.FileIO(
            f'gs://{params.BUCKET_NAME}/{params.BUCKET_TRAIN_DATA_PATH}/X_train',
            'w'), X)
    #making test and training set with train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train = expand_dims(X_train, axis=-1)
    X_test = expand_dims(X_test, axis=-1)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()
    print(len(X_train))
