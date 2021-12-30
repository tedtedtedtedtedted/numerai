from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from utils import *
from halo import Halo

# Download the required data
download_data()

# Load the downloaded training and validation datasets into dask DataFrames
training_set = read_data("data/numerai_training_data_int8.parquet")
validation_set = read_data("data/numerai_validation_data_int8.parquet")

X_train, y_train, X_val, y_val = \
    extract_minimum_learning_data(training_set, validation_set)

# LGBM model
################################################################################
# Compute prediction
params = {"n_estimators": 30000,
          "learning_rate": 0.1,
          "max_depth": 6,
          "num_leaves": 2 ** 6,
          "colsample_bytree": 0.1}

model = LGBMRegressor(**params)

spinner = Halo(text='', spinner='dots')
spinner.start('Training model')
model.fit(X_train, y_train)
spinner.succeed()

# Compute prediction
spinner.start('Making prediction')
y_pred = model.predict(X_val)
spinner.succeed()

# Evaluate accuracy
print('LightGBM model accuracy score: {0:0.4f}'.format(
    mean_squared_error(y_val, y_pred)))
label = validation_set.index.values

# Export to csv
make_csv(label, y_pred, model_name=str(model))

