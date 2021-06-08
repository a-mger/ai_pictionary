from google.cloud import storage
from ai_pictionary import params
from ai_pictionary.data import get_data
from ai_pictionary.cnn import initialize_model
from tensorflow.keras.utils import to_categorical
import os


class Trainer(object):
    def __init__(self, X, y, labels):
        """
            X: numpy array
            y: numpy array
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.labels = labels

    def run(self):
        self.pipeline = initialize_model(self.X, self.labels)
        history = self.pipeline.fit(self.X,
                                    self.y,
                                    batch_size=128,
                                    epochs=5,
                                    validation_split=0.3,
                                    verbose=1)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return y_pred


def upload_model_to_gcp():
    client = storage.Client()
    bucket = client.bucket(params.BUCKET_NAME)
    blob = bucket.blob(params.STORAGE_LOCATION)
    blob.upload_from_filename('model.save')


def save_model(reg):
    """method that saves the model into a .save file and uploads it on Google Storage /models folder"""
    SAVE_PATH = os.path.join("gs://", params.BUCKET_NAME , params.STORAGE_LOCATION)
    reg.save(SAVE_PATH)
    print("saved model.joblib in cloud")
    pass


if __name__ == "__main__":
    # get training data from GCP bucket
    X_train, X_test, y_train, y_test, labels = get_data(
        max_items_per_class=10000, max_labels=50)
    #normalize the data by dividing it by 255
    X_train /= 255
    X_test /= 255
    y_train_cat = to_categorical(y_train, len(labels))
    y_test_cat = to_categorical(y_test, len(labels))
    print("data preprocessed")
    # # Train and save model
    trainer = Trainer(X=X_train, y=y_train_cat, labels = labels)
    print("call trainer")
    reg = trainer.run()
    print("data trained")
    save_model(reg)
    print("data saved")
