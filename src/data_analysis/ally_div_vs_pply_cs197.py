import matplotlib.pyplot as plt
import numpy as np

model_with_ce_dict = {}

# Add name of your model here and the cross entropy losses in the order USPTO, PubMed, and USPTO+PubMed
model_with_ce_dict['GPT2-117M-2B'] = [5.862, 5.822, 5.708]


diversity_levels = ['USPTO', 'PubMed', 'USPTO+PubMed']
x_positions = {'USPTO': 0.158, 'PubMed': 0.168, 'USPTO+PubMed': 0.174}

colors = ['red', 'green', 'blue', 'orange', 'purple']

# Plotting
plt.figure(figsize=(20, 12))
for i, (model, ces) in enumerate(model_with_ce_dict.items()):
  plt.scatter([x_positions[level] for level in diversity_levels], ces, label=model, marker='o', color=colors[i])
  plt.plot([x_positions[level] for level in diversity_levels], ces, color=colors[i])
  
# Adding labeled vertical lines for USPTO, PubMed, and USPTO+PubMed
plt.axvline(x=0.158, color='black', label='')
plt.annotate('USPTO', xy=(0.158, plt.ylim()[0]), xytext=(20, 10), textcoords='offset points', ha='center', va='center')
plt.axvline(x=0.168, color='black', label='')
plt.annotate('PubMed', xy=(0.168, plt.ylim()[0]), xytext=(20, 10), textcoords='offset points', ha='center', va='center')
plt.axvline(x=0.174, color='black', label='')
plt.annotate('USPTO+PubMed', xy=(0.174, plt.ylim()[0]), xytext=(20, 10), textcoords='offset points', ha='center', va='center')


plt.title('Cross-Entropy Loss of Models Evaluated on Formally Diverse Datasets')
plt.xlabel('Task2Vec Diversity Coefficient of Training Dataset')
plt.ylabel('Cross-Entropy Loss Evaluated on OpenWebText2')


# Display the plot
plt.grid(True)
plt.legend()
plt.show()