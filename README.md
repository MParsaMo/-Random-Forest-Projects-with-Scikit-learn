🐍 Random Forest Projects with Scikit-learn
This repository contains three different machine learning projects using the Random Forest Classifier from Scikit-learn. Each project applies classification techniques on a different dataset, including:

credit_data.csv (credit default classification)

Digits Dataset (handwritten digit recognition with hyperparameter tuning)

Iris Dataset (basic flower species classification)

📦 Dependencies
All scripts use Python and require the following libraries:
pip install numpy pandas scikit-learn

📁 Project Structure
1. Credit Default Classification
File: credit_rf_cross_validation.py
Goal: Predict whether a customer will default based on income, age, and loan.

Steps:

Loads CSV data (credit_data.csv)

Selects three features: income, age, and loan

Uses RandomForestClassifier with 200 trees

Applies 10-fold cross-validation to evaluate model performance

Key Output:

Average test score across all folds
print(np.mean(prediction['test_score']))

2. Digit Recognition with Hyperparameter Tuning
File: digit_rf_gridsearch.py
Goal: Classify handwritten digits (0-9) from the Scikit-learn digits dataset.

Steps:

Flattens the 8×8 images into 64-feature vectors

Splits into training and test sets

Uses GridSearchCV to tune hyperparameters:

n_estimators, max_depth, min_samples_leaf

Trains the best model on training data

Evaluates it on test data

Key Output:

Best hyperparameters

Confusion matrix

Accuracy score

3. Iris Flower Classification
File: iris_rf_classifier.py
Goal: Classify iris flowers into three species using petal and sepal measurements.

Steps:

Loads Iris dataset from Scikit-learn

Splits data into 80/20 train/test sets

Trains a Random Forest with 1000 trees and max_features='sqrt'

Predicts and evaluates on test set

Key Output:

Confusion matrix

Accuracy score

🔍 Summary
| Project               | Dataset           | Model Type                 | Evaluation                  |
| --------------------- | ----------------- | -------------------------- | --------------------------- |
| Credit Classification | `credit_data.csv` | Random Forest + CV         | Cross-validation mean score |
| Digit Recognition     | Digits dataset    | Random Forest + GridSearch | Accuracy + Confusion Matrix |
| Iris Classification   | Iris dataset      | Random Forest              | Accuracy + Confusion Matrix |


✅ How to Run
Each Python file is self-contained. Run them individually:
python RandomForestCreditData.py
python RandomForestDigits.py
python RandomForestIris.py

📌 Notes
GridSearchCV may take longer due to extensive hyperparameter combinations.

You can reduce n_estimators or folds in cross-validation for faster testing.

All three examples use supervised classification with labeled datasets.

👨‍💻 Author
Developed as part of machine learning practice using Scikit-learn and Random Forests.
