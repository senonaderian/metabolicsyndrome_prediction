import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
import tensorflow as tf

# Read the CSV file into a DataFrame
df = pd.read_csv('3.csv')

# Extract the 'metabolicsyndrome' column and store it in a separate variable
target = df['metabolicsyndrome']

# Select all columns except for the 'metabolicsyndrome' column
data = df.drop('metabolicsyndrome', axis=1)

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)

# Apply a transformation to make the data non-negative
data_non_negative = data_imputed - np.min(data_imputed) + 1e-6

# Create a SelectKBest object using chi2 score function and fit to data
selector_chi2 = SelectKBest(chi2, k=10)
X_chi2 = selector_chi2.fit_transform(data_non_negative, target)

# Print selected features using chi2 score function
print("Selected Features (using chi-square score function):")
selected_features_chi2 = data.columns[selector_chi2.get_support()].tolist()
print(selected_features_chi2)

# Convert selected features to DataFrame
selected_features_df = pd.DataFrame({'selected_features': selected_features_chi2})

# Save the selected features to a file
selected_features_df.to_csv('selected_features6.csv', index=False)

# Load the dataset
selected_features = pd.read_csv("selected_features6.csv")

# Split the data into features (X) and target (y)
X = data[selected_features['selected_features']]
y = target

# Compute class weights
class_counts = np.bincount(y)
total_samples = np.sum(class_counts)
class_weights = total_samples / (len(class_counts) * np.where(class_counts != 0, class_counts, 1e-6))

# Define the custom loss function with class weighting
def weighted_loss(y_true, y_pred):
    weights = class_weights[y_true]
    loss = tf.losses.binary_crossentropy(y_true, y_pred, from_logits=True)
    weighted_loss = tf.reduce_mean(loss * weights)
    return weighted_loss

# Create a Multi-Layer Perceptron (Neural Network) classifier
mlp = MLPClassifier(random_state=42, max_iter=1500, learning_rate_init=0.001, hidden_layer_sizes=(100, 50))

# Set the custom loss function
mlp.loss_ = weighted_loss

# Train the MLP classifier on the features
mlp.fit(X, y)

# Evaluate the classifier
y_pred = mlp.predict(X)
cm = confusion_matrix(y, y_pred, labels=[1, 2])
tn, fp, fn, tp = cm.ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
f1 = f1_score(y, y_pred, average="weighted")
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp) if tn + fp != 0 else np.nan

print("Accuracy:", accuracy)
print("F1 score:", f1)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
