from pathlib import Path
import dask.dataframe as dd
import numerapi
import pandas as pd
from dask.dataframe import DataFrame
import json
from halo import Halo



def download_data() -> None:
    """Downloads the latest Numerai training, validation, and live data under
    ./data
    """
    # Create instance of NumerAPI
    napi = numerapi.NumerAPI()

    # Get the current round
    CURRENT_ROUND = napi.get_current_round()

    # Check all files if they are parquet and int8. If so, download it. We use
    # the int8 datasets instead of floats to reduce the computing power required
    # in training
    for file in napi.list_datasets():
        if "parquet" in file and "int8" in file:
            if "training" in file or "validation" in file:
                napi.download_dataset(file, f"data/{file}")
            else:
                Path(f"data/{CURRENT_ROUND}").mkdir(exist_ok=True, parents=True)
                napi.download_dataset(file, f"data/{CURRENT_ROUND}/{file}")


def read_data(path: str) -> DataFrame:
    """Read data stored in .parquet files as a dask DataFrame

    :param path: path to the parquet file
    :return: the data in the parquet file as a DataFrame
    """
    df = dd.read_parquet(f'{path}')
    return df

def extract_minimum_learning_data(training_set, validation_set) -> \
        (DataFrame, DataFrame, DataFrame, DataFrame):
    """Extract minimum learning data from the provided training set and
    validation set

    :param training_set: Training dataset
    :param validation_set: Validation dataset
    :return: (training features, training targets,
    validation features, validation targets)
    """
    # Isolate all feature and target names
    print('Extracting minimal training data...')
    # read the feature metadata amd get the "small" feature set
    with open("features.json", "r") as f:
        feature_metadata = json.load(f)
    features = feature_metadata["feature_sets"]["small"]

    # Training data
    X_train = training_set[features]
    y_train = training_set['target_nomi_20']

    # Validation data
    X_val = validation_set[features]
    y_val = validation_set['target_nomi_20']

    print('Done!')

    return X_train, y_train, X_val, y_val


def make_csv(labels, predictions, model_name='model') -> None:
    """ Export predictions as a csv file in the format wanted by numerai
    """
    print('Exporting Predictions to csv...')
    data_frame = pd.DataFrame(data=predictions, index=labels, columns=["prediction"])
    data_frame.index.name = 'id'
    data_frame.to_csv(f'{model_name}_prediction.csv')
    print('Done!')
