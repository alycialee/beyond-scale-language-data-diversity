"""
https://chat.openai.com/g/g-KV0CvoH8Y-python-excellent-comments-doc-strings-types/c/2df71842-3ce5-4e26-990c-b54e62d8cd50
"""
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import numpy as np  # Import numpy for numerical operations

# Dictionaries to store cross-entropy losses for models on different datasets
owt2_dict: dict[str, list[float]] = {}
c4_dict: dict[str, list[float]] = {}

# Adding model names and their cross-entropy losses for different datasets
owt2_dict['GPT2-51M-550M'] = [6.47, 6.36, 6.20]
owt2_dict['GPT2-117M-2.2B'] = [5.862, 5.822, 5.708]
owt2_dict['GPT2-204M-1.3B'] = [6.16, 5.75, 5.60]
owt2_dict['GPT2-345M-2.2B'] = [5.668, 5.54, 5.46]
owt2_dict['GPT2-810M-2B'] = [5.59, 5.48, 5.40]
owt2_dict['GPT2-1.5B-180M'] = [7.28, 6.40, 6.45]
owt2_dict['LLaMA2-7B-Run1-6.36M'] = [7.99, 8.04, 7.67]
owt2_dict['LLaMA2-7B-Run5-6.36M'] = [8.1, 7.7, 7.65]
owt2_dict['LLaMA2-7B-Run6-6.36M'] = [8.00, 7.82, 7.76]

c4_dict['GPT2-51M-550M'] = [5.82, 5.85, 5.64]
c4_dict['GPT2-117M-2.2B'] = [5.29, 5.30, 5.15]
c4_dict['GPT2-204M-1.3B'] = [5.40, 5.24, 5.05]
c4_dict['GPT2-345M-2.2B'] = [5.06, 5.04, 4.90]
c4_dict['GPT2-810M-2B'] = [5.59, 5.48, 5.40]
c4_dict['GPT2-1.5B-180M'] = [7.28, 6.40, 6.45]
c4_dict['LLaMA2-7B-Run1-6.36M'] = [7.11, 7.12, 6.82]
c4_dict['LLaMA2-7B-Run5-6.36M'] = [7.31, 6.97, 6.87]
c4_dict['LLaMA2-7B-Run6-6.36M'] = [7.23, 7.09, 7.02]

# Diversity levels and their positions on the X-axis
diversity_levels: list[str] = ['USPTO', 'PubMed', 'USPTO+PubMed']
x_positions: dict[str, float] = {'USPTO': 0.158, 'PubMed': 0.168, 'USPTO+PubMed': 0.195}

# Colors for plotting
colors: list[str] = [
    'darkviolet', 'rebeccapurple', 'mediumslateblue', 'mediumblue', 'royalblue', 
    'deepskyblue', 'darkturquoise', 'darkcyan', 'aquamarine'
]

plt.figure(figsize=(22, 6))  # Set the figure size for the plots

# Function to plot data, fit lines, and calculate R^2 values
def plot_data_and_fit(subplot_position: int, data_dict: dict[str, list[float]], title: str, ylabel: str) -> None:
    plt.subplot(1, 2, subplot_position)  # Specify subplot position
    for i, (model, ces) in enumerate(data_dict.items()):
        # Scatter plot for each model
        x_vals = [x_positions[level] for level in diversity_levels]
        plt.scatter(x_vals, ces, label=model, marker='o', color=colors[i])
        # Fit a first-degree polynomial (line) and plot it
        coefficients = np.polyfit(x_vals, ces, 1)
        line_fit = np.poly1d(coefficients)
        plt.plot(x_vals, line_fit(x_vals), color=colors[i])
        # Calculate and display R^2 value
        correlation_matrix = np.corrcoef(ces, line_fit(x_vals))
        r_squared = correlation_matrix[0, 1]**2
        plt.annotate(f'$R^2 = {r_squared:.2f}$', xy=(x_vals[-1], line_fit(x_vals)[-1]), textcoords='offset points', xytext=(5,5))

    for level in diversity_levels:
        plt.axvline(x=x_positions[level], color='black', linestyle='dotted')  # Add vertical lines for diversity levels
        plt.annotate(level, xy=(x_positions[level], plt.ylim()[0]), xytext=(0, 10), textcoords='offset points', ha='center')

    plt.title(title)
    plt.xlabel('Task2Vec Diversity Coefficient of Training Dataset')
    plt.ylabel(ylabel)
    plt.grid(True)

# Plot data and fits for both datasets
plot_data_and_fit(1, owt2_dict, 'CE Loss Evaluated on OpenWebText2', 'Cross-Entropy Loss Evaluated on OpenWebText2')
plot_data_and_fit(2, c4_dict, 'CE Loss Evaluated on C4', 'Cross-Entropy Loss Evaluated on C4')

plt.tight_layout()  # Adjust layout to make room for the legend
plt.savefig('./div_vs_val_ce_with_r2.png')  # Save the figure
plt.show()  # Show the plot
