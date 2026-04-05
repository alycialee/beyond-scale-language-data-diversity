import matplotlib.pyplot as plt
import numpy as np

owt2_dict = {}
c4_dict = {}

# Add the name of your model here and the cross-entropy losses in the order USPTO, PubMed, and USPTO+PubMed
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

diversity_levels = ['USPTO', 'PubMed', 'USPTO+PubMed']
x_positions = {'USPTO': 0.158, 'PubMed': 0.168, 'USPTO+PubMed': 0.195}

# colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'brown', 'gray']
colors = ['darkviolet', 'rebeccapurple', 'mediumslateblue', 'mediumblue', 'royalblue', 'deepskyblue', 'darkturquoise', 'darkcyan', 'aquamarine']

# Create two subplots side by side
plt.figure(figsize=(22, 6))

# First subplot
plt.subplot(1, 2, 1)
for i, (model, ces) in enumerate(owt2_dict.items()):
    plt.scatter([x_positions[level] for level in diversity_levels], ces, label=model, marker='o', color=colors[i])
    coefficients = np.polyfit([x_positions[level] for level in diversity_levels], ces, 1)  # Fit a first-degree polynomial (line)
    line_fit = np.poly1d(coefficients)  # Create a polynomial function
    plt.plot([x_positions[level] for level in diversity_levels], line_fit([x_positions[level] for level in diversity_levels]), color=colors[i])

plt.axvline(x=0.158, color='black', label='USPTO', linestyle='dotted')
plt.annotate('USPTO', xy=(0.158, plt.ylim()[0]), xytext=(20, 8), textcoords='offset points', ha='center', va='center')
plt.axvline(x=0.168, color='black', label='PubMed', linestyle='dotted')
plt.annotate('PubMed', xy=(0.168, plt.ylim()[0]), xytext=(22, 8), textcoords='offset points', ha='center', va='center')
plt.axvline(x=0.195, color='black', label='USPTO+PubMed', linestyle='dotted')
plt.annotate('USPTO+PubMed', xy=(0.195, plt.ylim()[0]), xytext=(-45, 8), textcoords='offset points', ha='center', va='center')

plt.title('CE Loss Evaluated on OpenWebText2')
plt.xlabel('Task2Vec Diversity Coefficient of Training Dataset')
plt.ylabel('Cross-Entropy Loss Evaluated on OpenWebText2')
plt.grid(True)
legend1 = plt.legend(loc='upper left', bbox_to_anchor=(-.36, 1))   # Place legend outside and to the left of first plot
plt.gca().add_artist(legend1)  # Add the first legend to the current Axes

# Second subplot
plt.subplot(1, 2, 2)
for i, (model, ces) in enumerate(c4_dict.items()):
    plt.scatter([x_positions[level] for level in diversity_levels], ces, label=model, marker='o', color=colors[i])
    coefficients = np.polyfit([x_positions[level] for level in diversity_levels], ces, 1)  # Fit a first-degree polynomial (line)
    line_fit = np.poly1d(coefficients)  # Create a polynomial function
    plt.plot([x_positions[level] for level in diversity_levels], line_fit([x_positions[level] for level in diversity_levels]), color=colors[i])

plt.axvline(x=0.158, color='black', label='USPTO', linestyle='dotted')
plt.annotate('USPTO', xy=(0.158, plt.ylim()[0]), xytext=(20, 8), textcoords='offset points', ha='center', va='center')
plt.axvline(x=0.168, color='black', label='PubMed', linestyle='dotted')
plt.annotate('PubMed', xy=(0.168, plt.ylim()[0]), xytext=(22, 8), textcoords='offset points', ha='center', va='center')
plt.axvline(x=0.195, color='black', label='USPTO+PubMed', linestyle='dotted')
plt.annotate('USPTO+PubMed', xy=(0.195, plt.ylim()[0]), xytext=(-45, 8), textcoords='offset points', ha='center', va='center')


plt.title('CE Loss Evaluated on C4')
plt.xlabel('Task2Vec Diversity Coefficient of Training Dataset')
plt.ylabel('Cross-Entropy Loss Evaluated on C4')
plt.grid(True)

# plt.suptitle('Cross-Entropy Loss of Models Evaluated on Datasets of High Diversity', fontsize=16)  # Set a centered title for the entire figure
# plt.savefig('./testfig.png', dpi=300)
plt.savefig('./div_vs_val_ce.png')

plt.show()