import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Create sample dataset
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Train a CART tree (uses Gini impurity)
tree = DecisionTreeClassifier(criterion='gini', max_depth=3)
tree.fit(X, y)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('CART Algorithm Visualization (scikit-learn gini)', fontsize=16, y=0.95)

# Plot 1: Original Data
axes[0, 0].scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
axes[0, 0].set_title('Original Data')
axes[0, 0].set_xlabel('Feature 1')
axes[0, 0].set_ylabel('Feature 2')

# Plot 2: Decision Boundaries
xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 100),
                     np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 100))
Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axes[0, 1].contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
axes[0, 1].scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
axes[0, 1].set_title('Decision Boundaries')
axes[0, 1].set_xlabel('Feature 1')
axes[0, 1].set_ylabel('Feature 2')

# Plot 3: Decision Tree Structure
plot_tree(tree, feature_names=['X1', 'X2'],
          class_names=['Class 0', 'Class 1'],
          filled=True, ax=axes[1, 0])
axes[1, 0].set_title('Decision Tree Structure')

# Plot 4: Text explanation
axes[1, 1].axis('off')
explanation = """
How CART Works:

- Uses Gini impurity to split nodes
- Can handle both classification and regression
- Creates binary splits

This visualization uses 'gini' impurity.

Blue/Red regions: classification after training.
Tree: decision rules used for classification.
"""
axes[1, 1].text(0.1, 0.5, explanation, fontsize=12, va='center')

# Save the plot
plt.tight_layout()
plt.savefig('cart_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualization saved as 'cart_visualization.png'")