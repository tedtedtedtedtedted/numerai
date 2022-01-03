# Functionalize the official model's (LGBM's) script.
# Light Gradient Boost Model Machine (LGBM) model.


import time
from lightgbm import LGBMRegressor
import gc

from numerapi import NumerAPI
from halo import Halo
from utils import *


class LGBMModel:
    """
    Light Gradient Boost Model Machine (LGBM) model.
    """

    def __init__(self) -> None:
        """
        Initialize and prepare data used in this model.
        """

        self.spinner = Halo(text='', spinner='dots')
        self.num_feature_neutralization = 50  # parameter for feature neutralization used after we make our predictions
        self.params = {"n_estimators": 50000,
                       "learning_rate": 0.00075,
                       "max_depth": 6,
                       "num_leaves": 2 ** 6,
                       "colsample_bytree": 0.5,
                       "subsample": 0.5,
                       "max_bins": 255,
                       "device": "gpu"}
        self.model_name = ("lgbm" + str(self.params["n_estimators"]) +
                           "-" + str(self.params["learning_rate"])
                           + "-" + str(self.params["colsample_bytree"])
                           + "-" + str(self.params["subsample"])
                           + "-" + f"neutral{self.num_feature_neutralization}").replace(
            ".", "dot")
        self.model = LGBMRegressor(**self.params)

        # Read the feature metadata and get the "small" feature set.
        with open("features.json", "r") as f:
            feature_metadata = json.load(f)
        self.features = feature_metadata["feature_sets"]["small"]

        # Read the training and validation data given the predefined features
        # stored in parquets as pandas DataFrames.
        self.training_data, self.validation_data = read_learning_data(self.features)

        # Extract feature matrix and target vector used for training.
        self.training_sample = self.training_data.filter(like='feature_',
                                                         axis='columns')
        self.training_target = self.training_data[TARGET_COL]

        # Extract feature matrix and target vector used for validation.
        self.validation_sample = self.validation_data.filter(like='feature_',
                                                             axis='columns')
        self.validation_target = self.validation_data[TARGET_COL]

        # TODO: Decide using gc.collect() here?

    def train_model(self) -> None:
        """
        Train the model.
        """

        self.spinner.start('Training model')
        self.model.fit(self.training_sample, self.training_target)
        self.spinner.succeed()

    def predict(self) -> np.ndarray[float]:
        """
        Predict.
        """
        self.spinner.start('Predicting on validation data')
        self.validation_data.loc[:, f"preds_{self.model_name}"] = \
            self.model.predict(self.training_sample)
        self.spinner.succeed()

    def neutralize_features(self) -> None:
        """
        Neutralize features.
        """

        self.spinner.start('Neutralizing to risky features')
        neutralize_riskiest_features(self.training_data, self.validation_data,
                                     self.features, self.model_name,
                                     k=self.num_feature_neutralization)
        self.spinner.succeed()

    def export_predictions(self) -> None:
        """
        Export predictions to in CSV format.
        """
        self.spinner.start('Exporting Predictions to csv...')
        self.validation_data["prediction"] = self.validation_data[f"preds_{self.model_name}_neutral_riskiest_{self.num_feature_neutralization}"] \
            .rank(pct=True)
        self.validation_data["prediction"].to_csv(f"validation_predictions_{self.model_name}.csv")
        self.spinner.succeed()

