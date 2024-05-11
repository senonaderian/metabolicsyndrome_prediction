import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer

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
from sklearn.feature_selection import SelectKBest, f_classif
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
clf = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X, y)
clf_optimized = grid_search.best_estimator_

# Evaluate the classifier using cross-validation
scores = cross_val_score(clf_optimized, X, y, cv=5)

# Print the accuracy and F1 score
print("Accuracy (f_classif):", scores.mean())
print("F1 score (f_classif):", f1_score(y, clf_optimized.predict(X), average="weighted"))

# Calculate the confusion matrix
y_pred = clf_optimized.predict(X)
cm = confusion_matrix(y, y_pred, labels=[0, 1])  # Specify all possible labels
if cm.size >= 4:
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print("Sensitivity (f_classif):", sensitivity)
    print("Specificity (f_classif):", specificity)
else:
    print("Insufficient values in the confusion matrix.")

# Plot the decision tree with improved aesthetics
plt.figure(figsize=(15, 10))
plot_tree(clf_optimized, filled=True, feature_names=X.columns.tolist(), class_names=["1", "2"], max_depth=3)
plt.savefig("decision_tree.png", dpi=200)
plt.show()
