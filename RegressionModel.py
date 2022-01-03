# An abstract superclass for many regression models.
import numpy as np

from utils import *


class RegressionModel:
    """
    An abstract superclass for many regression models.


    """

    def __init__(self) -> None:
        self.x_train = None
        self.x_val = None
        self.y_train = None
        self.y_val = None

    def update_data(self) -> bool:
        """
        Receive/update data used for training the model.
        TODO: Decide whether store the data inside this model or just update
              references (of training, testing, validation data).
        """

        try:
            # Download the required data
            download_data()

            # Load the downloaded training and validation datasets into dask DataFrames
            training_set = read_data("data/numerai_training_data_int8.parquet")
            validation_set = read_data("data/numerai_validation_data_int8.parquet")

            self.x_train, self.y_train, self.x_val, self.y_val = \
                extract_minimum_learning_data(training_set, validation_set)
        except() # TODO: Determine the signature of "download_data()" and determine whether there'd be exceptions thrown.


    def train_model(self) -> bool:
        """
        Train the model.
        """
        pass

    def predict(self) -> np.ndarray[float]:
        """
        Predict
        """
        pass

