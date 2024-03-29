from google.cloud import storage
from ai_pictionary import params
from ai_pictionary.data import get_data, get_data2
from ai_pictionary.cnn import initialize_model
from tensorflow.keras.utils import to_categorical
import os
from tensorflow.keras.callbacks import EarlyStopping


class Trainer(object):
    def __init__(self, X, y, labels):
        """
            X: numpy array
            y: numpy array
        """
        self.model = None
        self.X = X
        self.y = y
        self.labels = labels

    def run(self):
        """fit model"""
        self.model = initialize_model(self.X, self.labels)
        es = EarlyStopping(patience=2, restore_best_weights=True)
        history = self.model.fit(self.X,
                                 self.y,
                                 batch_size=128,
                                 epochs=30,
                                 validation_split=0.2,
                                 verbose=1)
        return self.model


def save_model(reg):
    """method that saves the model and uploads it on Google Storage /models folder"""
    SAVE_PATH = os.path.join("gs://", params.BUCKET_NAME , params.STORAGE_LOCATION)
    reg.save(SAVE_PATH)
    print("saved model in cloud")
    pass


if __name__ == "__main__":
    # get training data from GCP bucket
    X_train, X_test, y_train, y_test, labels = get_data(
        max_items_per_class=50000, max_labels=50)
    #normalize the data by dividing it by 255
    X_train /= 255
    X_test /= 255
    y_train_cat = to_categorical(y_train, len(labels))
    y_test_cat = to_categorical(y_test, len(labels))
    # # Train and save model
    trainer = Trainer(X=X_train, y=y_train_cat, labels = labels)
    reg = trainer.run()
    save_model(reg)
