"""

"""
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
import numpy as np

# -- x-axis: diversity coefficient
names_for_div_coeff = ['USPTO', 'PubMed', 'USPOT + PubMed']
div_coeff = [0.1580, 0.1681, 0.1742]

# -- y-axis: perplexity (ppl), the the eval data sets below
pile_all_subsets = [788.7322, 805.0728, 585.6262]
openwebtext2_pile = [520.8103, 470.7946, 436.1974]

# -- Compute R^2 values
# Fit a linear model to your data points
slope, intercept = np.polyfit(div_coeff, pile_all_subsets, 1)
# Predict the y-values using the Linear Model
linear_pile_all_subsets = [slope * xi + intercept for xi in div_coeff]
r2_pile_all_subsets = r2_score(pile_all_subsets, linear_pile_all_subsets)

# Fit a linear model to your data points
slope, intercept = np.polyfit(div_coeff, openwebtext2_pile, 1)
# Predict the y-values using the Linear Model
linear_openwebtext2_pile = [slope * xi + intercept for xi in div_coeff]
r2_openwebtext2_pile = r2_score(openwebtext2_pile, linear_openwebtext2_pile)

# Plotting
plt.figure(figsize=(10,6))

# Plotting linear fits
plt.plot(div_coeff, linear_pile_all_subsets, 'r-', label=f'Linear Fit Pile All Subsets (R^2={r2_pile_all_subsets:.2f})')
plt.plot(div_coeff, linear_openwebtext2_pile, 'b-', label=f'Linear Fit OpenWebText2 Pile (R^2={r2_openwebtext2_pile:.2f})')

# Plotting scatter points
plt.scatter(div_coeff, pile_all_subsets, marker='o', color='r', label='Pile All Subsets')
plt.scatter(div_coeff, openwebtext2_pile, marker='s', color='b', label='OpenWebText2 Pile')

# Adding title and labels
plt.title('Diversity Coefficient (div coeff) vs Perplexity (ppl)')
plt.xlabel('Diversity Coefficient (div coeff)')
plt.ylabel('Perplexity (ppl)')
plt.legend()

# Save the plot
desktop_path = os.path.expanduser("./")
plt.savefig(os.path.join(desktop_path, 'div_coeff_vs_ppl_linear_fit.pdf'))
plt.savefig(os.path.join(desktop_path, 'div_coeff_vs_ppl_linear_fit.png'))
plt.savefig(os.path.join(desktop_path, 'div_coeff_vs_ppl_linear_fit.svg'))

# Show the plot
plt.show()


