# Ensembling Methods in Machine Learning
# Ensembling in machine learning refers to the technique of combining predictions from multiple models to produce a more robust and accurate result than any single model could achieve alone. The main idea is that by aggregating the strengths of different models, the ensemble can reduce errors due to bias, variance, or noise.

# There are several popular ensembling methods, including bagging, boosting, stacking, and voting. Bagging (Bootstrap Aggregating) involves training multiple versions of a model on different random subsets of the data and then averaging their predictions, as seen in Random Forests. Boosting builds models sequentially, where each new model focuses on correcting the errors of the previous ones; examples include AdaBoost and Gradient Boosting Machines. Stacking, or stacked generalization, combines the predictions of several base models using a meta-model, which learns how best to blend their outputs. Voting is a simpler approach where multiple models vote on the final prediction, and the majority or average is taken as the ensemble output.

# Ensembling is widely used in machine learning competitions and real-world applications because it often leads to significant improvements in predictive performance, especially when the individual models are diverse and make different types of errors. However, ensembles can be more computationally expensive and harder to interpret than single models.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Base Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Ensemble Methods
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.ensemble import StackingClassifier

# Load and prepare data
data = load_breast_cancer()
X, y = data.data, data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary to store results
results = {}

def evaluate_model(model, name, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    return accuracy

### 1. Voting Classifier
print("\n### Voting Classifier ###")
clf1 = LogisticRegression(random_state=42)
clf2 = DecisionTreeClassifier(random_state=42)
clf3 = SVC(probability=True, random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', clf1), ('dt', clf2), ('svc', clf3)],
    voting='soft'
)
voting_acc = evaluate_model(voting_clf, 'Voting', X_train_scaled, X_test_scaled, y_train, y_test)
print(f"Voting Classifier Accuracy: {voting_acc:.4f}")

### 2. Bagging
print("\n### Bagging ###")
bagging = BaggingClassifier(
    DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    random_state=42
)
bagging_acc = evaluate_model(bagging, 'Bagging', X_train_scaled, X_test_scaled, y_train, y_test)
print(f"Bagging Classifier Accuracy: {bagging_acc:.4f}")

### 3. Random Forest
print("\n### Random Forest ###")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_acc = evaluate_model(rf, 'Random Forest', X_train_scaled, X_test_scaled, y_train, y_test)
print(f"Random Forest Accuracy: {rf_acc:.4f}")

### 4. AdaBoost
print("\n### AdaBoost ###")
ada = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    random_state=42
)
ada_acc = evaluate_model(ada, 'AdaBoost', X_train_scaled, X_test_scaled, y_train, y_test)
print(f"AdaBoost Accuracy: {ada_acc:.4f}")

### 5. Gradient Boosting
print("\n### Gradient Boosting ###")
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_acc = evaluate_model(gb, 'Gradient Boosting', X_train_scaled, X_test_scaled, y_train, y_test)
print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")

### 6. Stacking
print("\n### Stacking ###")
estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('svc', SVC(probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)
stacking_acc = evaluate_model(stacking, 'Stacking', X_train_scaled, X_test_scaled, y_train, y_test)
print(f"Stacking Classifier Accuracy: {stacking_acc:.4f}")

# Visualize results
plt.figure(figsize=(12, 6))
methods = list(results.keys())
accuracies = list(results.values())

colors = sns.color_palette("husl", len(methods))
bars = plt.bar(methods, accuracies, color=colors)
plt.title('Comparison of Ensemble Methods')
plt.xlabel('Method')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig('ensembling_ensemble_comparison.png')
plt.close()

# Feature importance plot for Random Forest
plt.figure(figsize=(12, 6))
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.title('Feature Importances (Random Forest)')
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), data.feature_names[indices], rotation=90)
plt.tight_layout()
plt.savefig('ensembling_feature_importance.png')
plt.close()

print("\nFiles created:")
print("ensembling_ensemble_comparison.png")
print("ensembling_feature_importance.png")