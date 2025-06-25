import numpy as np
import pandas as pd # Used for better confusion matrix display
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import datasets # To load the digits dataset
import matplotlib.pyplot as plt # For potential data visualization (e.g., sample images)

def load_digits_data():
    """
    Loads the handwritten digits dataset from scikit-learn.

    This dataset consists of 1797 8x8 pixel grayscale images of handwritten digits (0-9).
    Each image is a numerical representation of an integer (0-9).

    Returns:
        sklearn.utils.Bunch: A scikit-learn Bunch object containing data, target,
                             images, and description.
    """
    print("Loading handwritten digits dataset...")
    digits = datasets.load_digits()

    print("\n--- Digits Data Overview ---")
    print(f"Number of samples: {len(digits.images)}")
    print(f"Image shape: {digits.images[0].shape} (8x8 pixels)")
    print(f"Total features per image (after flattening): {digits.images[0].size}")
    print("Target classes (digits):", np.unique(digits.target))
    print("\nFirst sample (image and target):")
    # Display the first image as an example
    plt.figure(figsize=(2, 2))
    plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(f"Target: {digits.target[0]}")
    plt.axis('off')
    plt.show(block=False) # Use block=False to prevent blocking script execution immediately
    plt.pause(0.5) # Short pause to display the plot
    plt.close() # Close the figure

    return digits

def preprocess_data(digits_dataset):
    """
    Preprocesses the digits data by reshaping the 2D image arrays into 1D feature vectors.

    Machine learning models typically expect input features to be in a 2D array
    where each row is a sample and each column is a feature.

    Args:
        digits_dataset (sklearn.utils.Bunch): The loaded digits dataset.

    Returns:
        tuple: A tuple containing (features (X), target (y)).
    """
    print("\n--- Preprocessing Data ---")
    # Reshape each 8x8 image into a one-dimensional vector of 64 features.
    # `len(digits_dataset.images)` is the number of samples.
    # `-1` tells NumPy to automatically calculate the size of the second dimension
    # based on the total number of elements. So, each image becomes a row of 64 pixel values.
    image_features = digits_dataset.images.reshape((len(digits_dataset.images), -1))
    image_target = digits_dataset.target

    print(f"Original image shape: {digits_dataset.images[0].shape}")
    print(f"Reshaped features shape: {image_features.shape}") # Should be (n_samples, 64)
    return image_features, image_target

def split_dataset(features, target, train_size=0.8, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Args:
        features (numpy.ndarray): The preprocessed feature data.
        target (numpy.ndarray): The target labels.
        train_size (float): The proportion of the dataset to include in the train split.
        random_state (int): Controls the shuffling applied to the data before splitting.
                            Ensures reproducibility.

    Returns:
        tuple: A tuple containing (features_train, features_test, target_train, target_test).
    """
    print(f"\nSplitting data into training ({train_size*100:.0f}%) and testing ({(1-train_size)*100:.0f}%) sets...")
    # `stratify=target` ensures that the proportion of classes in the training and testing sets
    # is roughly the same as in the original dataset. This is important for classification tasks.
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, train_size=train_size, random_state=random_state, stratify=target
    )
    print(f"Training set size: {len(features_train)} samples")
    print(f"Testing set size: {len(len(features_test))} samples")
    return features_train, features_test, target_train, target_test

def tune_random_forest_hyperparameters(features_train, target_train, random_state=42, cv_folds=10):
    """
    Performs GridSearchCV to find the optimal hyperparameters for the Random Forest model.

    Args:
        features_train (numpy.ndarray): Training features.
        target_train (numpy.ndarray): Training target.
        random_state (int): Seed for reproducibility of the Random Forest.
        cv_folds (int): Number of folds for cross-validation within GridSearchCV.

    Returns:
        sklearn.model_selection.GridSearchCV: The fitted GridSearchCV object,
                                              containing the best estimator.
    """
    print("\n--- Tuning Random Forest Hyperparameters using GridSearchCV ---")
    # Initialize Random Forest Classifier with some default parameters.
    # n_jobs=-1 uses all available CPU cores for parallel processing, speeding up GridSearchCV.
    # max_features='sqrt' is a common default for Random Forests, meaning each tree considers
    # sqrt(n_features) randomly selected features when looking for the best split.
    random_forest_model = RandomForestClassifier(n_jobs=-1, max_features='sqrt', random_state=random_state)

    # Define the hyperparameter grid for GridSearchCV
    param_grid = {
        # n_estimators: The number of trees in the forest. More trees generally improve accuracy
        #               but increase computation time.
        "n_estimators": [10, 100, 500, 1000],
        # max_depth: The maximum depth of each decision tree in the forest. Controls overfitting.
        #            A deeper tree can model more complex relationships but might overfit.
        "max_depth": [1, 5, 10, 15, None], # None means nodes are expanded until all leaves are pure
        # min_samples_leaf: The minimum number of samples required to be at a leaf node.
        #                   Increasing this value can smooth the model, avoiding overfitting.
        "min_samples_leaf": [1, 2, 4, 10] # Reduced the range for quicker execution
    }

    # Perform grid search with cross-validation
    # estimator: The model to tune (RandomForestClassifier)
    # param_grid: Dictionary of parameter names and values to test
    # cv: Number of folds for cross-validation for parameter tuning (e.g., 10-fold CV)
    # n_jobs=-1: Use all available CPU cores for parallel execution of GridSearchCV.
    # verbose: Controls the verbosity of the output.
    # scoring: The metric used to evaluate each model during cross-validation.
    grid_search = GridSearchCV(
        estimator=random_forest_model,
        param_grid=param_grid,
        cv=cv_folds,
        n_jobs=-1,
        verbose=2, # Increased verbose to see more details during grid search
        scoring='accuracy',
        refit=True # After search, refit the best model on the entire training data
    )

    # Fit the grid search to the training data
    grid_search.fit(features_train, target_train)

    print("\n--- GridSearchCV Results ---")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation score (accuracy): {grid_search.best_score_:.4f}")
    print(f"Best estimator (trained model): {grid_search.best_estimator_}")

    return grid_search

def evaluate_model(fitted_model, features_test, target_test, target_names):
    """
    Evaluates the trained Random Forest model (best estimator from GridSearchCV)
    on the test data and prints performance metrics.

    Args:
        fitted_model (sklearn.ensemble.RandomForestClassifier): The trained RF model.
        features_test (numpy.ndarray): Testing features.
        target_test (numpy.ndarray): Testing target (true labels).
        target_names (list): List of class names for better confusion matrix readability.
    """
    print("\n--- Evaluating Model Performance on Test Set ---")
    # Make predictions on the test dataset using the best model found
    grid_predictions = fitted_model.predict(features_test)

    # Confusion Matrix: A table used to describe the performance of a classification model.
    # Rows represent true classes, columns represent predicted classes.
    print('\nConfusion Matrix:')
    cm = confusion_matrix(target_test, grid_predictions)
    # Convert to DataFrame for better readability with target names
    cm_df = pd.DataFrame(cm, index=[f'Actual {name}' for name in target_names],
                         columns=[f'Predicted {name}' for name in target_names])
    print(cm_df)

    # Accuracy Score: The proportion of correctly classified instances.
    accuracy = accuracy_score(target_test, grid_predictions)
    print(f'\nAccuracy Score: {accuracy:.4f}')

    # Classification Report: Provides precision, recall, f1-score for each class.
    print("\nClassification Report:")
    print(classification_report(target_test, grid_predictions, target_names=target_names))

if __name__ == "__main__":
    # Define parameters for reproducibility and split size
    TRAIN_TEST_SPLIT_RATIO = 0.8
    RANDOM_SEED = 42
    GRID_SEARCH_CV_FOLDS = 5 # Reduced from 10 for potentially faster execution during demonstration

    # 1. Load the Digits Dataset
    digits_dataset = load_digits_data()

    # 2. Preprocess Data (Flatten images into feature vectors)
    image_features, image_target = preprocess_data(digits_dataset)

    # 3. Split Data into Training and Testing Sets
    image_features_train, image_features_test, image_target_train, image_target_test = split_dataset(
        image_features, image_target,
        train_size=TRAIN_TEST_SPLIT_RATIO,
        random_state=RANDOM_SEED
    )

    # 4. Tune Random Forest Hyperparameters using GridSearchCV
    grid_search_result = tune_random_forest_hyperparameters(
        image_features_train, image_target_train,
        random_state=RANDOM_SEED,
        cv_folds=GRID_SEARCH_CV_FOLDS
    )

    # The best_estimator_ attribute of GridSearchCV is the best model, already fitted on the training data.
    best_rf_model = grid_search_result.best_estimator_

    # 5. Evaluate the Best Model on the Held-Out Test Set
    evaluate_model(best_rf_model, image_features_test, image_target_test, digits_dataset.target_names)

    print("\nScript execution complete.")
