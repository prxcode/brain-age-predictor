import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function to load data (metadata and brain activity matrices)
def load_data(metadata_file, data_dir):
    # Load metadata
    metadata = pd.read_csv(metadata_file)

    # Load brain activity matrices
    brain_data = []
    for file in os.listdir(data_dir):
        if file.endswith(".tsv"):  # Assuming brain data is in TSV format
            matrix = pd.read_csv(os.path.join(data_dir, file), sep="\t").values
            brain_data.append(matrix)
    
    return brain_data, metadata

# Preprocess data (flatten brain activity matrices)
def preprocess_data(brain_data, metadata):
    # Flatten brain activity matrices
    flattened_data = [matrix.flatten() for matrix in brain_data]
    X = np.array(flattened_data)

    # Use age as the target variable
    y = metadata['age'].values

    return X, y

# Train the model using RandomForestRegressor
def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print("Validation MAE:", mean_absolute_error(y_val, y_pred))
    return model

# Predict ages on the test data
def predict_ages(model, test_data_dir):
    test_brain_data = []
    for file in os.listdir(test_data_dir):
        if file.endswith(".tsv"):
            matrix = pd.read_csv(os.path.join(test_data_dir, file), sep="\t").values
            test_brain_data.append(matrix.flatten())  # Flatten the test data matrices

    test_X = np.array(test_brain_data)
    return model.predict(test_X)

def main():
    # Set relative paths
    project_root = os.path.dirname(os.path.abspath(__file__))  # Get the current script's directory
    data_dir = os.path.join(project_root, '..', 'data')  # Assuming data is in the parent directory of the script

    train_metadata_file = os.path.join(data_dir, 'metadata', 'training_metadata.csv')
    test_metadata_file = os.path.join(data_dir, 'metadata', 'test_metadata.csv')
    train_data_dir = os.path.join(data_dir, 'train_tsv')
    test_data_dir = os.path.join(data_dir, 'test_tsv')

    # Load and preprocess training data
    brain_data, metadata = load_data(train_metadata_file, train_data_dir)
    X, y = preprocess_data(brain_data, metadata)

    # Train the model
    model = train_model(X, y)

    # Predict on test data
    predictions = predict_ages(model, test_data_dir)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()
