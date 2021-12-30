import xgboost as xgb
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

# XGBoost model
################################################################################
# Compute prediction
params = {"objective": "reg:logistic",
          'n_estimators': 2000,
          'learning_rate': 0.1,
          'max_depth': 8,
          'alpha': 0.3,
          }
xgb_rgs = xgb.XGBRegressor(**params)

spinner = Halo(text='', spinner='dots')
spinner.start('Training model')
xgb_rgs.fit(X_train, y_train)
spinner.succeed()

# Compute prediction
spinner.start('Making prediction')
y_pred = xgb_rgs.predict(X_val)
spinner.succeed()

# Evaluate accuracy
print('XGBoost model accuracy score: {0:0.4f}'.format(
    mean_squared_error(y_val, y_pred)))
label = validation_set.index.values

# Export to csv
make_csv(label, y_pred, model_name='xgboost')

