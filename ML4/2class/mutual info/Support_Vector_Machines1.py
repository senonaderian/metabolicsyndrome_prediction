import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Read the CSV file into a DataFrame
df = pd.read_csv('ex2.csv')

# Extract the 'metabolicsyndrome' column and store it in a separate variable
target = df['metabolicsyndrome']

# Select all columns except for the 'metabolicsyndrome' column
data = df.drop('metabolicsyndrome', axis=1)

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)

# Apply a transformation to make the data non-negative
data_non_negative = data_imputed - np.min(data_imputed) + 1e-6

# Create a SelectKBest object using mutual_info_classif score function and fit to data
selector_mutual_info = SelectKBest(mutual_info_classif, k=10)
X_chi2 = selector_mutual_info.fit_transform(data_non_negative, target)

# Print selected features using mutual_info_classif score function
print("Selected Features (using mutual_info_classif score function):")
selected_features_mi = data.columns[selector_mutual_info.get_support()].tolist()
print(selected_features_mi)

# Convert selected features to DataFrame
selected_features_df = pd.DataFrame({'selected_features': selected_features_mi})

# Save the selected features to a file
selected_features_df.to_csv('selected_features6.csv', index=False)

# Load the dataset
selected_features = pd.read_csv("selected_features6.csv")

# Split the data into features (X) and target (y)
X = data[selected_features['selected_features']]
y = target

# Train an SVM classifier without class weighting
clf = SVC(random_state=42)

# Fit the SVM classifier on the entire dataset
clf.fit(X, y)

# Evaluate the classifier using cross-validation
scores = cross_val_score(clf, X, y, cv=5)

# Print the accuracy and F1 score using cross-validation
print("Accuracy:", scores.mean())
print("F1 score:", f1_score(y, clf.predict(X), average="weighted"))

# Calculate the confusion matrix
y_pred = clf.predict(X)
cm = confusion_matrix(y, y_pred, labels=[0, 1])  # Specify all possible labels
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp) if tn + fp != 0 else np.nan

print("Sensitivity:", sensitivity)
print("Specificity:", specificity)