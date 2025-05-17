import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Create sample dataset
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Train a C4.5-like tree (uses entropy)
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3)
tree.fit(X, y)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('C4.5 Algorithm Visualization (scikit-learn entropy)', fontsize=16, y=0.95)

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
How C4.5 Works:

- Like ID3, but uses Information Gain Ratio
- Handles continuous features and missing values
- Prunes tree to avoid overfitting

This visualization uses 'entropy' (info gain)
as a stand-in for C4.5's gain ratio.

Blue/Red regions: classification after training.
Tree: decision rules used for classification.
"""
axes[1, 1].text(0.1, 0.5, explanation, fontsize=12, va='center')

# Save the plot
plt.tight_layout()
plt.savefig('c45_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualization saved as 'c45_visualization.png'")