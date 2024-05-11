import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Read the CSV file into a DataFrame
df1 = pd.read_csv('ex3.csv')
df2 = pd.read_csv('ex2.csv')

# Extract the 'metabolicsyndrome' column and store it in a separate variable
target1 = df1['metabolicsyndrome']
target2 = df2['metabolicsyndrome']

# Select all columns except for the 'metabolicsyndrome' column
data1 = df1.drop('metabolicsyndrome', axis=1)
data2 = df2.drop('metabolicsyndrome', axis=1)

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
data_imputed1 = imputer.fit_transform(data1)
data_imputed2 = imputer.fit_transform(data2)

# Apply a transformation to make the data non-negative
data_non_negative1 = data_imputed1 - np.min(data_imputed1) + 1e-6
data_non_negative2 = data_imputed2 - np.min(data_imputed2) + 1e-6

# Feature selection for each dataset
selector_mi1 = SelectKBest(mutual_info_classif, k=10)
X_mi1 = selector_mi1.fit_transform(data_non_negative1, target1)

selector_mi2 = SelectKBest(mutual_info_classif, k=10)
X_mi2 = selector_mi2.fit_transform(data_non_negative2, target2)

# Define classifiers
knn = KNeighborsClassifier(n_neighbors=3)
clf_rf = RandomForestClassifier(criterion="entropy", random_state=42)
clf_svm = SVC(random_state=42)

# Create pipelines for preprocessing and classification
knn_pipe = make_pipeline(StandardScaler(), knn)
rf_pipe = make_pipeline(StandardScaler(), clf_rf)
svm_pipe = make_pipeline(StandardScaler(), clf_svm)

# Define the voting classifier
voting_clf = VotingClassifier(estimators=[('knn', knn_pipe), ('rf', rf_pipe), ('svm', svm_pipe)], voting='hard')

# Perform cross-validation and evaluate the ensemble
for X, target, name in [(X_mi1, target1, 'Dataset 1'), (X_mi2, target2, 'Dataset 2')]:
    scores = cross_val_score(voting_clf, X, target, cv=5)
    print(f"{name}:")
    print("Accuracy:", scores.mean())
    print("F1 score:", f1_score(target, voting_clf.fit(X, target).predict(X), average="weighted"))
    y_pred = voting_clf.fit(X, target).predict(X)
    cm = confusion_matrix(target, y_pred, labels=[0, 1])  # Specify all possible labels
    if cm.size >= 4:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity)
    else:
        print("Insufficient values in the confusion matrix.")
