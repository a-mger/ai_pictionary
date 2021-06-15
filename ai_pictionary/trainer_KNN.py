from google.cloud import storage
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from ai_pictionary import params
from ai_pictionary.data import get_data, get_data_KNN
from ai_pictionary.cnn import initialize_model
from tensorflow.keras.utils import to_categorical
import os
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import io
from tensorflow.python.lib.io import file_io

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
        self.model = initialize_model(self.X, self.labels)
        es = EarlyStopping(patience=2, restore_best_weights=True)
        history = self.model.fit(self.X,
                                 self.y,
                                 batch_size=128,
                                 epochs=30,
                                 validation_split=0.2,
                                 verbose=1)
        return self.model
    
    def run_KNN(self):
        self.model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        self.model.fit(self.X, self.y)
        results_knn = cross_val_score(self.model, self.X, self.y).mean()
        return self.model, results_knn

    def evaluate(self, X_test, y_test):
        """evaluates the nodel on df_test and return the RMSE"""
        y_pred = self.model.predict(X_test)
        return y_pred


def save_model(reg):
    """method that saves the model into a .save file and uploads it on Google Storage /models folder"""
    SAVE_PATH = os.path.join("gs://", params.BUCKET_NAME , params.STORAGE_LOCATION)
    reg.save(SAVE_PATH)
    print("saved model in cloud")
    pass

def save_model_KNN(reg):
    """method that saves the model into a .save file and uploads it on Google Storage /models folder"""
    SAVE_PATH = file_io.FileIO(
            f'gs://{params.BUCKET_NAME}/{params.STORAGE_LOCATION}/KNN.joblib',
            'w')
    joblib.dump(reg, SAVE_PATH)
    pass

if __name__ == "__main__":
    # get training data from GCP bucket
    #X_train, X_test, y_train, y_test, labels = get_data(
        #max_items_per_class=90000, max_labels=250)
    #X_train, y_train, labels = get_data2()
    #normalize the data by dividing it by 255
    X_train_KNN, y_train_KNN, labels = get_data_KNN(
        max_items_per_class=1000, max_labels=5)
    X_train_KNN /= 255
    #X_train /= 255
    #X_test /= 255
    #y_train_cat = to_categorical(y_train, len(labels))
    y_train_cat_KNN = to_categorical(y_train_KNN, len(labels))
    #y_test_cat = to_categorical(y_test, len(labels))
    print("data preprocessed")
    # # Train and save model
    #trainer = Trainer(X=X_train, y=y_train_cat, labels = labels)
    trainer = Trainer(X=X_train_KNN, y=y_train_cat_KNN, labels = labels)
    print("call trainer")
    reg, results_knn = trainer.run_KNN()
    print("data trained")
    #save_model(reg)
    print("data saved")
    print(results_knn)
