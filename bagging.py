# bagging_real_world.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score

sns.set_style("whitegrid")
np.random.seed(42)

# ---------- 1. CLASSIFICATION: IRIS ----------
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# Reduce to 2-D with PCA so we can draw a decision boundary
X_iris_2d = PCA(n_components=2).fit_transform(X_iris)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_iris_2d, y_iris, test_size=0.2, random_state=42, stratify=y_iris)

bag_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42
)
bag_clf.fit(X_train_c, y_train_c)
y_pred_c = bag_clf.predict(X_test_c)
acc = accuracy_score(y_test_c, y_pred_c)
print(f"Iris ‑ Bagging accuracy: {acc:.3f}")

# --- 1a  Decision-boundary plot ---
xmin, xmax = X_iris_2d[:,0].min()-1, X_iris_2d[:,0].max()+1
ymin, ymax = X_iris_2d[:,1].min()-1, X_iris_2d[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(xmin, xmax, 400),
                     np.linspace(ymin, ymax, 400))
ZZ = bag_clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(6,5))
plt.contourf(xx, yy, ZZ, alpha=0.25, cmap="Pastel2")
sns.scatterplot(x=X_train_c[:,0], y=X_train_c[:,1], hue=y_train_c,
                palette="Set1", edgecolor="k", s=40, alpha=0.9)
plt.title("Bagging decision regions – Iris (PCA 2-D)")
plt.xlabel("PC 1"); plt.ylabel("PC 2")
plt.tight_layout()
plt.savefig("bagging_iris_decision_boundary.png")
plt.close()

# --- 1b  Confusion-matrix heat-map ---
cm = confusion_matrix(y_test_c, y_pred_c)
plt.figure(figsize=(4,3.5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title("Iris – Confusion Matrix")
plt.ylabel("True"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("bagging_iris_confusion_matrix.png")
plt.close()

# ---------- 2. REGRESSION: CALIFORNIA HOUSING ----------
housing = fetch_california_housing()
X_h, y_h = housing.data, housing.target

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_h, y_h, test_size=0.2, random_state=42)

bag_reg = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=100,
    random_state=42
)
bag_reg.fit(X_train_r, y_train_r)
y_pred_r = bag_reg.predict(X_test_r)
mse = mean_squared_error(y_test_r, y_pred_r)
r2  = r2_score(y_test_r, y_pred_r)
print(f"California Housing ‑ Bagging MSE: {mse:.3f}")
print(f"California Housing ‑ Bagging R² : {r2:.3f}")

# --- 2a  True vs. Predicted scatter ---
plt.figure(figsize=(5,5))
plt.scatter(y_test_r, y_pred_r, alpha=0.3)
plt.plot([y_test_r.min(), y_test_r.max()],
         [y_test_r.min(), y_test_r.max()],
         'r--', lw=2, label="Ideal")
plt.xlabel("True Median House Value")
plt.ylabel("Predicted")
plt.title("Bagging Regressor – True vs Predicted")
plt.legend(); plt.tight_layout()
plt.savefig("bagging_housing_true_vs_pred.png")
plt.close()

# --- 2b  Residual distribution ---
residuals = y_test_r - y_pred_r
plt.figure(figsize=(5,3))
sns.histplot(residuals, bins=40, kde=True, color="teal")
plt.title("Residuals – Bagging Regressor")
plt.xlabel("True – Predicted"); plt.tight_layout()
plt.savefig("bagging_housing_residuals.png")
plt.close()

# Created/Modified files during execution:
for fname in [
    "iris_decision_boundary.png",
    "iris_confusion_matrix.png",
    "housing_true_vs_pred.png",
    "housing_residuals.png"
]:
    print(fname)

# Created/Modified files during execution:
# None (only displays plots in memory)


# Bagging, short for Bootstrap Aggregating, is a machine learning ensemble technique used to improve the stability and accuracy of machine learning algorithms used in classification and regression. It also reduces variance and helps to avoid overfitting.
# Bagging works by training multiple models (often of the same type) on different subsets of the training data and then aggregating their predictions. This approach can lead to a more robust model that generalizes better to unseen data.
# The main idea behind bagging is to create multiple versions of a predictor and use them to get an aggregated prediction. This is particularly useful when the model is prone to high variance, as it helps to smooth out the predictions.
# Bagging is a powerful ensemble method that can be applied to various base models, including decision trees, neural networks, and more. It is particularly effective when the base model is unstable, meaning that small changes in the training data can lead to large changes in the model's predictions.

# The process involves these key steps:

# Bootstrap Sampling: Multiple subsets are created from the original dataset through random sampling with replacement. This means some instances may appear multiple times in a subset, while others may be excluded.
# Model Training: A base machine learning algorithm (e.g., decision tree, neural network) is trained on each of these subsets. Each model learns from a slightly different perspective of the data.
# Aggregation: The predictions from all the models are combined. For classification, this is typically done by majority voting (the class predicted by the most models wins). For regression, the predictions are usually averaged.
# Bagging is particularly effective with "unstable" models, meaning those where small changes in the training data can cause significant changes in the model. Decision trees are a prime example of an unstable model, which is why bagging is often used in ensemble methods like Random Forests.