import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate # For robust model evaluation
import os # For checking file existence

def load_data(file_path):
    """
    Loads credit data from a CSV file into a Pandas DataFrame.
    If the file is not found, a dummy CSV is created for demonstration.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded DataFrame, or None if an error occurs
                          and a dummy file cannot be created.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        print("Please ensure 'credit_data.csv' is in the same directory as the script.")
        print("Creating a dummy 'credit_data.csv' for demonstration purposes.")
        # Create a dummy CSV for demonstration
        dummy_data = {
            'income': [50000, 60000, 30000, 70000, 45000, 55000, 65000, 25000, 80000, 40000,
                       52000, 48000, 75000, 32000, 68000, 28000, 58000, 72000, 38000, 62000],
            'age': [30, 45, 22, 50, 35, 28, 40, 60, 33, 25,
                    31, 44, 23, 51, 36, 29, 41, 61, 34, 26],
            'loan': [10000, 20000, 5000, 30000, 8000, 15000, 25000, 3000, 40000, 7000,
                     12000, 18000, 28000, 6000, 35000, 4000, 22000, 32000, 9000, 11000],
            'default': [0, 0, 1, 0, 0, 0, 0, 1, 0, 1,
                        0, 1, 0, 1, 0, 1, 0, 0, 1, 0] # 0 = no default, 1 = default
        }
        pd.DataFrame(dummy_data).to_csv(file_path, index=False)
        print("Dummy 'credit_data.csv' created. Please replace it with your actual data.")
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None

def prepare_features_target(dataframe, feature_cols, target_col):
    """
    Separates the DataFrame into features (X) and target (y) as NumPy arrays.

    Args:
        dataframe (pandas.DataFrame): The input DataFrame.
        feature_cols (list): A list of column names to be used as features.
        target_col (str): The name of the target column.

    Returns:
        tuple: A tuple containing (features (X), target (y)) as NumPy arrays.
    """
    # Convert features DataFrame and target Series to NumPy arrays
    # No need to reshape to (-1, 3) if features is already a 2D DataFrame.
    # .values converts the DataFrame/Series to a NumPy array.
    X = dataframe[feature_cols].values
    y = dataframe[target_col].values
    print("\n--- Prepared Data Shapes ---")
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    return X, y

def train_and_evaluate_random_forest(features, target, n_estimators=200, cv_folds=10, random_state=42):
    """
    Trains a Random Forest Classifier and evaluates its performance using cross-validation.

    Args:
        features (numpy.ndarray): The full feature dataset.
        target (numpy.ndarray): The full target dataset.
        n_estimators (int): The number of trees in the forest.
        cv_folds (int): The number of folds for cross-validation.
        random_state (int): Controls the randomness for reproducibility.

    Returns:
        dict: A dictionary of scores from cross_validate.
    """
    print(f"\n--- Training Random Forest with {n_estimators} estimators ---")
    print(f"--- Evaluating with {cv_folds}-Fold Cross-Validation ---")
    # Initialize Random Forest Classifier
    # n_estimators: The number of trees in the forest. More trees generally lead to better performance
    #               but also increase computation time.
    # random_state: Ensures reproducibility of the random forest's decision process.
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    # Perform cross-validation
    # cv: Defines the cross-validation splitting strategy.
    # scoring: Metric to evaluate the performance (accuracy for classification).
    # return_train_score: Set to False to only get test scores.
    prediction_results = cross_validate(
        estimator=model,
        X=features,
        y=target,
        cv=cv_folds,
        scoring='accuracy',
        return_train_score=False
    )

    print(f"Individual test scores for each fold: {prediction_results['test_score']}")
    mean_accuracy = np.mean(prediction_results['test_score'])
    std_accuracy = np.std(prediction_results['test_score'])
    print(f"Mean Cross-validation Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy*2:.4f})") # Mean +/- 2 std_dev

    return prediction_results

if __name__ == "__main__":
    CSV_FILE_PATH = 'credit_data.csv'
    FEATURE_COLUMNS = ['income', 'age', 'loan']
    TARGET_COLUMN = 'default'
    NUM_ESTIMATORS = 200 # Number of trees in the Random Forest
    CROSS_VALIDATION_FOLDS = 10 # Number of folds for cross-validation
    RANDOM_SEED = 42 # For reproducibility

    # 1. Load Data
    credit_data_df = load_data(CSV_FILE_PATH)
    if credit_data_df is None:
        exit() # Exit if data loading failed

    # 2. Prepare Features and Target
    X, y = prepare_features_target(credit_data_df, FEATURE_COLUMNS, TARGET_COLUMN)

    # 3. Train and Evaluate Random Forest Model using Cross-Validation
    cv_scores = train_and_evaluate_random_forest(
        X, y,
        n_estimators=NUM_ESTIMATORS,
        cv_folds=CROSS_VALIDATION_FOLDS,
        random_state=RANDOM_SEED
    )

    print("\nScript execution complete.")
