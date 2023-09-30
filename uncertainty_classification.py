#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Set the figure size for landscape ratio
plt.figure(figsize=(8.3, 5.8))

# Case 1: Well Separable Classes
# to demnstrate model uncertainty. different ways that the decision boundary
# can be drawn
np.random.seed(0)
class1 = np.random.randn(50, 2)
class2 = np.random.randn(50, 2) + [4, 3]

plt.scatter(class1[:, 0], class1[:, 1], c='k', marker='o', label='Class 1', edgecolors='k')
plt.scatter(class2[:, 0], class2[:, 1], c='grey', marker='x', label='Class 2', edgecolors='k')
# plt.title('Well Separable Classes', fontsize=12, loc='left', pad=30)
# plt.legend(loc='upper left', fontsize=10, bbox_to_anchor=(0.02, 0.98), bbox_transform=plt.gca().transAxes)

# Remove axes details
plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# Save the figure in black-and-white
plt.savefig('case1.png', dpi=300, bbox_inches='tight')
plt.show()


#%%
# Case 2: Overlapping Classes
# to demnstrate model uncertainty and still different ways that the decision 
# boundary can be drawn even
np.random.seed(1)
class1 = np.random.randn(50, 2)
class2 = np.random.randn(50, 2) + [2,2]

plt.figure(figsize=(8.3, 5.8))

plt.scatter(class1[:, 0], class1[:, 1], c='k', marker='o', label='Class 1', edgecolors='k')
plt.scatter(class2[:, 0], class2[:, 1], c='gray', marker='x', label='Class 2', edgecolors='k')
# plt.title('Overlapping Classes')
# plt.legend()

# Remove axes details
plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# Save the figure in black-and-white
plt.savefig('case2.png', dpi=300, bbox_inches='tight')
plt.show()

#%%

# Case 3: contoured 
# showing the data uncertainty by contouring the data points
# TODO: collission of the two classes not so necessary (could be a separate case of its own)
# 
np.random.seed(0)
class1 = np.random.randn(50, 2)
class2 = np.random.randn(50, 2) + [2, 2]

# Set up the figure
plt.figure(figsize=(8.3, 5.8))
plt.scatter(class1[:, 0], class1[:, 1], c='k', marker='o', label='Class 1', edgecolors='k')
plt.scatter(class2[:, 0], class2[:, 1], c='grey', marker='x', label='Class 2', edgecolors='k')

# Create a grid for contour plotting
x, y = np.meshgrid(np.linspace(-5, 8, 100), np.linspace(-5, 8, 100))
xy = np.vstack([x.ravel(), y.ravel()])

# Compute KDE for both classes
kernel_class1 = gaussian_kde(class1.T)
kernel_class2 = gaussian_kde(class2.T)
z1 = np.reshape(kernel_class1(xy).T, x.shape)
z2 = np.reshape(kernel_class2(xy).T, x.shape)


# Plot the contour of the KDE for Class 1
plt.contour(x, y, z1, levels=10, colors='b', alpha=0.5)

# Plot the contour of the KDE for Class 2
plt.contour(x, y, z2, levels=10, colors='r', alpha=0.5)

# Scatter plot for data points
# plt.scatter(class1[:, 0], class1[:, 1], c='b', marker='x', label='Class 1', linewidth=1)
# plt.scatter(class2[:, 0], class2[:, 1], c='r', marker='.', label='Class 2', linewidth=1)

# Title and legend placement
# plt.title('Well Separable Classes (High Variance)', fontsize=10, loc='left', pad=10)
# plt.legend(loc='upper left', fontsize=8, bbox_to_anchor=(0.02, 0.98), bbox_transform=plt.gca().transAxes)

# Remove axes details
plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# Save the figure in black-and-white
plt.savefig('case3.png', dpi=300, bbox_inches='tight')
plt.show()

#%%
# Case 4: contoured more samples
# showing the data uncertainty by contouring the data points
# denser (having more data). to make the point that more data cannot
# help with reducing the data uncertainty whereas it can help with the epistemic
# uncertainty. TODO: right now the example does not shed light on this matter. 
# It would be much better if we have a case where indeed the epistemic uncertainty 
# reduces with more data but then the data uncertainty remains more or less the same!
np.random.seed(0)
class1 = np.random.randn(500, 2)
class2 = np.random.randn(500, 2) + [2, 2]

# Set up the figure
plt.figure(figsize=(8.3, 5.8))
plt.scatter(class1[:, 0], class1[:, 1], c='k', marker='o', label='Class 1', edgecolors='k')
plt.scatter(class2[:, 0], class2[:, 1], c='grey', marker='x', label='Class 2', edgecolors='k')

# Create a grid for contour plotting
x, y = np.meshgrid(np.linspace(-5, 8, 100), np.linspace(-5, 8, 100))
xy = np.vstack([x.ravel(), y.ravel()])

# Compute KDE for both classes
kernel_class1 = gaussian_kde(class1.T)
kernel_class2 = gaussian_kde(class2.T)
z1 = np.reshape(kernel_class1(xy).T, x.shape)
z2 = np.reshape(kernel_class2(xy).T, x.shape)


# Plot the contour of the KDE for Class 1
plt.contour(x, y, z1, levels=10, colors='b', alpha=0.5)

# Plot the contour of the KDE for Class 2
plt.contour(x, y, z2, levels=10, colors='r', alpha=0.5)

# Scatter plot for data points
# plt.scatter(class1[:, 0], class1[:, 1], c='b', marker='x', label='Class 1', linewidth=1)
# plt.scatter(class2[:, 0], class2[:, 1], c='r', marker='.', label='Class 2', linewidth=1)

# Title and legend placement
# plt.title('Well Separable Classes (High Variance)', fontsize=10, loc='left', pad=10)
# plt.legend(loc='upper left', fontsize=8, bbox_to_anchor=(0.02, 0.98), bbox_transform=plt.gca().transAxes)

# Remove axes details
plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# Save the figure in black-and-white
plt.savefig('case3_moredata.png', dpi=300, bbox_inches='tight')
plt.show()
