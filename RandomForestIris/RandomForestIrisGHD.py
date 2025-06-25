import pandas as pd # Included for general data science context, and for readability of some outputs
import numpy as np # Used by scikit-learn internally
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import datasets # To load the Iris dataset

def load_iris_data():
    """
    Loads the Iris dataset from scikit-learn.

    The Iris dataset is a classic and very easy multi-class classification dataset.
    It consists of 150 samples of iris flowers, with 4 features each,
    and 3 possible target classes (species of iris).

    Returns:
        sklearn.utils.Bunch: A scikit-learn Bunch object containing data, target,
                             and target_names.
    """
    print("Loading Iris dataset...")
    iris_data = datasets.load_iris()

    # Print some information for better understanding
    print("\n--- Iris Data Overview ---")
    print(f"Features (data) shape: {iris_data.data.shape}") # (150 samples, 4 features)
    print(f"Target (labels) shape: {iris_data.target.shape}") # (150 samples,)
    print("Feature names:", iris_data.feature_names)
    print("Target names (species):", iris_data.target_names)
    print("\nFirst 5 rows of features:")
    print(iris_data.data[:5])
    print("\nFirst 5 target labels:")
    print(iris_data.target[:5])
    return iris_data

def prepare_features_target(iris_dataset):
    """
    Prepares the features (X) and target (y) arrays from the Iris dataset.

    Args:
        iris_dataset (sklearn.utils.Bunch): The loaded Iris dataset.

    Returns:
        tuple: A tuple containing (features (X), target (y)).
    """
    features = iris_dataset.data  # The features (e.g., sepal length, sepal width, etc.)
    target = iris_dataset.target # The target labels (species: 0, 1, 2)
    return features, target

def split_data(features, target, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Args:
        features (numpy.ndarray): The feature data.
        target (numpy.ndarray): The target data.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int, optional): Controls the shuffling applied to the data before splitting.
                                      Ensures reproducibility.

    Returns:
        tuple: A tuple containing (features_train, features_test, target_train, target_test).
    """
    print(f"\nSplitting data into training ({(1-test_size)*100:.0f}%) and testing ({test_size*100:.0f}%) sets...")
    # `stratify=target` ensures that the proportion of classes in the training and testing sets
    # is roughly the same as in the original dataset. This is crucial for classification tasks.
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=target
    )
    print(f"Training set size: {len(features_train)} samples")
    print(f"Testing set size: {len(features_test)} samples")
    return features_train, features_test, target_train, target_test

def train_random_forest_model(features_train, target_train, n_estimators=1000, max_features_val='sqrt', random_state=42):
    """
    Trains a Random Forest Classifier model.

    Args:
        features_train (numpy.ndarray): Training features.
        target_train (numpy.ndarray): Training target.
        n_estimators (int): The number of trees in the forest.
        max_features_val (str or float): The number of features to consider when looking for the best split.
                                         'sqrt' is common for classification.
        random_state (int): Controls the randomness for reproducibility.

    Returns:
        sklearn.ensemble.RandomForestClassifier: The trained Random Forest model.
    """
    print(f"\n--- Training Random Forest Model ---")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_features: {max_features_val}")
    model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features_val, random_state=random_state)
    model.fit(features_train, target_train)
    print("Random Forest model training complete.")
    return model

def evaluate_model(model, features_test, target_test, target_names):
    """
    Evaluates the trained Random Forest model on the test data and prints performance metrics.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): The trained Random Forest model.
        features_test (numpy.ndarray): Testing features.
        target_test (numpy.ndarray): Testing target (true labels).
        target_names (list): List of class names for better confusion matrix readability.
    """
    print("\n--- Evaluating Model Performance on Test Set ---")
    prediction = model.predict(features_test)

    # Confusion Matrix: A table used to describe the performance of a classification model.
    # Rows represent true classes, columns represent predicted classes.
    print('\nConfusion Matrix:')
    cm = confusion_matrix(target_test, prediction)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(cm_df)

    # Accuracy Score: The proportion of correctly classified instances.
    accuracy = accuracy_score(target_test, prediction)
    print(f'\nAccuracy Score: {accuracy:.4f}')

    # Classification Report: Provides precision, recall, f1-score for each class.
    print("\nClassification Report:")
    print(classification_report(target_test, prediction, target_names=target_names))

if __name__ == "__main__":
    # Define parameters for model and data splitting
    TEST_DATA_SPLIT_RATIO = 0.2
    RF_N_ESTIMATORS = 1000
    RF_MAX_FEATURES = 'sqrt'
    RANDOM_SEED = 42 # For reproducibility of splits and model training

    # 1. Load the Iris Dataset
    iris_data = load_iris_data()
    if iris_data is None:
        exit() # Should not happen for built-in datasets

    # 2. Prepare Features (X) and Target (y)
    X, y = prepare_features_target(iris_data)

    # 3. Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=TEST_DATA_SPLIT_RATIO,
        random_state=RANDOM_SEED
    )

    # 4. Train the Random Forest Model
    random_forest_classifier = train_random_forest_model(
        X_train, y_train,
        n_estimators=RF_N_ESTIMATORS,
        max_features_val=RF_MAX_FEATURES,
        random_state=RANDOM_SEED
    )

    # 5. Evaluate the Model on the Test Set
    evaluate_model(random_forest_classifier, X_test, y_test, iris_data.target_names)

    print("\nScript execution complete.")
