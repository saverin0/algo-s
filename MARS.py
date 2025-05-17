import numpy as np
import matplotlib.pyplot as plt

# 1. Generate sample data
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = np.sin(X).ravel() + 0.2 * np.random.randn(100)

def plot_step_by_step():
    """
    Visualize how MARS builds the model step by step
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Understanding MARS Algorithm Step by Step', fontsize=16)

    # Plot 1: Original Data
    axes[0,0].scatter(X, y, color='blue', alpha=0.5, label='Data points')
    axes[0,0].set_title('Step 1: Original Data')
    axes[0,0].set_xlabel('X')
    axes[0,0].set_ylabel('y')
    axes[0,0].legend()

    # Plot 2: Hinge Functions
    axes[0,1].scatter(X, y, color='blue', alpha=0.5, label='Data points')

    # Add some example hinge functions
    knot = 0
    h1 = np.maximum(0, X - knot)  # forward hinge
    h2 = np.maximum(0, knot - X)  # backward hinge

    axes[0,1].plot(X, h1, 'r-', label='max(0, X - knot)', linewidth=2)
    axes[0,1].plot(X, h2, 'g-', label='max(0, knot - X)', linewidth=2)
    axes[0,1].axvline(x=knot, color='k', linestyle='--', label='Knot')
    axes[0,1].set_title('Step 2: Hinge Functions')
    axes[0,1].set_xlabel('X')
    axes[0,1].set_ylabel('y')
    axes[0,1].legend()

    # Plot 3: Building the Model
    axes[1,0].scatter(X, y, color='blue', alpha=0.5, label='Data points')

    # Show how multiple hinge functions combine
    knots = [-1, 0, 1]
    colors = ['r', 'g', 'b']
    for knot, color in zip(knots, colors):
        h = np.maximum(0, X - knot)
        axes[1,0].plot(X, 0.5*h, color=color,
                      label=f'Hinge at x={knot}', alpha=0.5)
        axes[1,0].axvline(x=knot, color=color,
                         linestyle='--', alpha=0.3)

    axes[1,0].set_title('Step 3: Multiple Hinge Functions')
    axes[1,0].set_xlabel('X')
    axes[1,0].set_ylabel('y')
    axes[1,0].legend()

    # Plot 4: Final Model
    axes[1,1].scatter(X, y, color='blue', alpha=0.5, label='Data points')

    # Simulate a MARS-like fit
    y_pred = 0
    for knot in knots:
        h1 = np.maximum(0, X - knot)
        h2 = np.maximum(0, knot - X)
        y_pred += 0.5 * h1 - 0.3 * h2

    axes[1,1].plot(X, y_pred, 'r-', label='MARS fit', linewidth=2)
    axes[1,1].set_title('Step 4: Final MARS Model')
    axes[1,1].set_xlabel('X')
    axes[1,1].set_ylabel('y')
    axes[1,1].legend()

    plt.tight_layout()
    plt.savefig('mars_explanation.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create the visualization
plot_step_by_step()

print("""
MARS Algorithm Steps Explanation:

1. Start with data points (top-left plot)
   - MARS begins with your raw data

2. Create hinge functions (top-right plot)
   - These are the building blocks of MARS
   - Each hinge function splits data at a 'knot'
   - Two types: forward (max(0, X-knot)) and backward (max(0, knot-X))

3. Build model iteratively (bottom-left plot)
   - MARS adds hinge functions one at a time
   - Each new function improves the fit
   - Knots are placed at strategic points

4. Final model (bottom-right plot)
   - Combines multiple hinge functions
   - Creates a flexible, piecewise linear fit
   - Adapts to local patterns in data

Key Features:
- Automatically finds important variables
- Determines optimal locations for knots
- Creates non-linear relationships
- Handles interactions between variables
""")