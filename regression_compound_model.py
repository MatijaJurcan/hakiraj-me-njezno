import pandas as pd
import os
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


# Value depending on where project data is stored relative to script directory
RELATIVE_PATH_TO_CSV_DATA = 'data'
RELATIVE_PATH_TO_CSV_OUTPUT = 'output'
CURRENT_DIR = os.getcwd()
CSV_DATA_DIR = os.path.join(CURRENT_DIR, RELATIVE_PATH_TO_CSV_DATA)
CSV_OUTPUT_DIR = os.path.join(CURRENT_DIR, RELATIVE_PATH_TO_CSV_OUTPUT)

# Initialize compound model - dictionary that keeps all models - result of training
compound_model = {}

train_file_name = 'kaggle_x_train.csv'
result_file_name = 'kaggle_y_train.csv'
testing_file_name = 'kaggle_x_test.csv'
result_test_file_name = 'kaggle_y_test_final.csv'

# Save file path from current dir
train_file_path = os.path.join(CSV_DATA_DIR, train_file_name)
result_file_path = os.path.join(CSV_DATA_DIR, result_file_name)
testing_file_path = os.path.join(CSV_DATA_DIR, testing_file_name)
result_test_file_path = os.path.join(CSV_OUTPUT_DIR, result_test_file_name)

# Import data from csv to dataframes
X_train_df = pd.read_csv(train_file_path)
y_train_df = pd.read_csv(result_file_path)
X_test_df = pd.read_csv(testing_file_path)

# Extract y training dataset header
y_header = y_train_df.columns.tolist()

######################## START MODEL TRAINING ########################
# Manipulate training X dataframe - get location as string based on boolean values and drop unnecessary columns afterwards
X_train_df['location'] = X_train_df.apply(lambda row: 'quadriceps' if row['is_quadriceps']
                                          else 'hamstring' if row['is_hamstring']
                                          else 'calf' if row['is_calf']
                                          else 'add-abd' if row['is_add_abd']
                                          else 'belly' if row['is_belly']
                                          else 'other', axis=1)
columns_to_drop = ['is_contact',
                   'tone',
                   'palpation',
                   'is_quadriceps',
                   'is_hamstring',
                   'is_calf',
                   'is_add_abd',
                   'is_belly'
                   ]
X_train_df = X_train_df.drop(columns=columns_to_drop)

# Find max value of injury duration as limit for prediction - totally arbitrary
y_max = max(y_train_df['injury_duration'])
y_min = min(y_train_df['injury_duration'])

# Group X training data by combination of class and location attributes into single training groups for different regression models into splits of X training set
split_X_train_dfs = {}
grouped_dfs = X_train_df.groupby(['class', 'location'])

for group, group_df in grouped_dfs:
    # Group key value is based on all possible combinations of class and location value in data
    class_val, location_val = group

    # Removing no longer necessary location columns
    columns_to_remove = ['location', 'class']
    group_df = group_df.drop(columns=columns_to_remove)

    # Split dataset are saved as dicts with key as name
    split_X_train_dfs[(class_val, location_val)] = group_df


for key, split_X_train_df in split_X_train_dfs.items():
    # Get all unique index values inside split of base X training dataset and filter y training dataset by those ids
    index_values = split_X_train_df['Id'].unique()
    filtered_y_train_df = y_train_df[y_train_df['Id'].isin(index_values)]

    # Remove id column from X split of training set and filtered y training results
    split_X_train_df = split_X_train_df.drop('Id', axis=1)
    filtered_y_train_df = filtered_y_train_df.drop('Id', axis=1)

    # Split whole training set into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        split_X_train_df, filtered_y_train_df, test_size=0.2)

    # Initialize instances of models from scikit-learn and initialize mse scoring dictionaries and saved_models dictionary
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'ElasticNet Regression': ElasticNet(),
        'Random Forest Regression': RandomForestRegressor(),
        'Gradient Boosting Regression': GradientBoostingRegressor()
    }
    mse_scores = {}
    saved_models = {}

    # For each regression model, fit based on X training values and y training values
    for model_name, model in models.items():
        model.fit(X_train.values, y_train.values.ravel())

        # Calculate model prediction for x training test values from previous split, also calculate mean squared error
        y_pred = model.predict(X_test.values)
        mse = mean_squared_error(y_test, y_pred)

        # Save both mse score and model into respective dictionaries
        mse_scores[model_name] = mse
        saved_models[model_name] = model
    # Select model with minimum mean square error and save the best regression model to the compound_model of whole problem.
    best_model = min(mse_scores, key=mse_scores.get)
    best_mse = mse_scores[best_model]
    compound_model[key] = saved_models[best_model]
######################## END MODEL TRAINING ########################


######################## START MODEL TESTING########################
# # Frst step is finding location of each entry based on boolean and turning it into string representation.
X_test_df['location'] = X_test_df.apply(lambda row: 'quadriceps' if row['is_quadriceps']
                                        else 'hamstring' if row['is_hamstring']
                                        else 'calf' if row['is_calf']
                                        else 'add-abd' if row['is_add_abd']
                                        else 'belly' if row['is_belly']
                                        else 'other', axis=1)

# Removing unnecessary columns from X_test dataframe and initialize y_predictions_df dataframe
X_test_df = X_test_df.drop(columns=columns_to_drop)

y_predictions_df = pd.DataFrame(columns=y_header)

# Iterate over each data point inside X_test dataframe
for index, data_point in X_test_df.iterrows():

    # Find class and location values from data point
    test_class_value = data_point['class']
    test_location_value = data_point['location']

    # Set key name of model as test_model_name and based on that key select appropriate mini-model from compound model
    test_model_name = (test_class_value, test_location_value)
    test_model = compound_model[test_model_name]
    y_id = data_point['Id']

    # Remove unnecessary columns from data point for predicting value of y
    data_point = data_point.drop(['Id', 'location', 'class'])

    # Predict y value using model
    y_pred = test_model.predict(data_point.values.reshape(1, -1))

    # Account for injury duration minimum and maximum values observed in training data set and if value exceeds that - set it to the closest boundary value
    if y_pred[0] < y_min:
        y_pred[0] = y_min
    elif y_pred[0] > y_max:
        y_pred[0] = y_max

    # Calculate rounded value to ceiling and create row list with id and value
    y_value = math.ceil(y_pred[0])
    y_row = [y_id, y_value]

    # Add row list to dataframe at specified location
    y_predictions_df.loc[len(y_predictions_df)] = y_row

# Load results array into dataframe for output and the output it to the file
y_predictions_df.to_csv(result_test_file_path, index=False)

######################## END MODEL TESTING########################
