import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

# Read the CSV file into a DataFrame
df = pd.read_csv('ex3.csv')

# Extract the 'metabolicsyndrome' column and store it in a separate variable
target = df['metabolicsyndrome']

# Select all columns except for the 'metabolicsyndrome' column
data = df.drop('metabolicsyndrome', axis=1)

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)

# Apply a transformation to make the data non-negative
data_non_negative = data_imputed - np.min(data_imputed) + 1e-6

# Create a SelectKBest object using f_classif score function and fit to data
selector_f_classif = SelectKBest(f_classif, k=10)
X_f_classif = selector_f_classif.fit_transform(data_non_negative, target)

# Print selected features using f_classif score function
print("Selected Features (using f_classif score function):")
selected_features_f_classif = data.columns[selector_f_classif.get_support()].tolist()
print(selected_features_f_classif)

# Convert selected features to DataFrame
selected_features_df = pd.DataFrame({'selected_features': selected_features_f_classif})

# Save the selected features to a file
selected_features_df.to_csv('selected_features1.csv', index=False)

# Load the dataset
selected_features = pd.read_csv("selected_features1.csv")

# Split the data into features (X) and target (y)
X = data[selected_features['selected_features']]
y = target

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Train a decision tree classifier with hyperparameter tuning
clf_dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(clf_dt, param_grid, cv=5)
grid_search.fit(X, y)
clf_optimized_dt = grid_search.best_estimator_

# Train a Gaussian Naive Bayes classifier
clf_nb = GaussianNB()

# Ensemble classifiers using voting
voting_clf = VotingClassifier(estimators=[('dt', clf_optimized_dt), ('nb', clf_nb)], voting='soft')

# Evaluate the ensemble classifier using cross-validation
scores = cross_val_score(voting_clf, X, y, cv=5, scoring='accuracy')
f1_scores = cross_val_score(voting_clf, X, y, cv=5, scoring='f1_weighted')

# Print the accuracy and F1 score
print("Accuracy (Ensemble):", scores.mean())
print("F1 score (Ensemble):", f1_scores.mean())

# Calculate the confusion matrix
y_pred = voting_clf.predict(X)
cm = confusion_matrix(y, y_pred, labels=[0, 1])  # Specify all possible labels
if cm.size >= 4:
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print("Sensitivity (Ensemble):", sensitivity)
    print("Specificity (Ensemble):", specificity)
else:
    print("Insufficient values in the confusion matrix.")

# No need to plot the decision tree since it's part of an ensemble
