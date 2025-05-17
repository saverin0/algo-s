import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Create sample dataset
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('ID3 Algorithm Visualization', fontsize=16, y=0.95)

# Plot 1: Original Data
axes[0, 0].scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
axes[0, 0].set_title('Original Data')
axes[0, 0].set_xlabel('Feature 1')
axes[0, 0].set_ylabel('Feature 2')

# Plot 2: Decision Boundaries
tree = DecisionTreeClassifier(max_depth=3)
tree.fit(X, y)

# Create mesh grid for decision boundary
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
How ID3 Works:

1. Start with all data
2. Find best feature to split on
3. Create decision boundary
4. Repeat for subsets until:
   - Max depth reached
   - Pure subset achieved
   - No more features

Blue/Red regions show
how data is classified
after training.

Tree structure shows
the decision rules used
for classification.
"""
axes[1, 1].text(0.1, 0.5, explanation, fontsize=12, va='center')

# Save the plot
plt.tight_layout()
plt.savefig('id3_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualization saved as 'id3_visualization.png'")