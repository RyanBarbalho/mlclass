import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the abalone dataset from the CSV file
abalone_df = pd.read_csv("abalone_dataset.csv")

# Set the style of the visualization
sns.set(style="whitegrid")

# 1. Scatter plot for length vs diameter colored by type
plt.figure(figsize=(10, 6))
sns.scatterplot(x='length', y='diameter', hue='type', data=abalone_df, palette='viridis')
plt.title('Length vs Diameter by Type')
plt.xlabel('Length')
plt.ylabel('Diameter')
plt.legend(title='Type')
plt.savefig('length_vs_diameter_by_type.png')
plt.close()

# 2. Scatter plot for height vs length colored by type
plt.figure(figsize=(10, 6))
sns.scatterplot(x='height', y='length', hue='type', data=abalone_df, palette='viridis')
plt.title('Height vs Length by Type')
plt.xlabel('Height')
plt.ylabel('Length')
plt.legend(title='Type')
plt.savefig('height_vs_length_by_type.png')
plt.close()

# 3. Pair plot for a subset of the features colored by type
subset = abalone_df[['length', 'diameter', 'height', 'whole_weight', 'type']]
sns.pairplot(subset, hue='type', palette='viridis')
plt.savefig('pairplot_by_type.png')
plt.close()

# 4. Box plot for whole_weight by type
plt.figure(figsize=(10, 6))
sns.boxplot(x='type', y='whole_weight', data=abalone_df, palette='viridis')
plt.title('Whole Weight by Type')
plt.xlabel('Type')
plt.ylabel('Whole Weight')
plt.savefig('whole_weight_by_type.png')
plt.close()

# 4. Scatter plot for whole weight vs shucked weight colored by type
plt.figure(figsize=(10, 6))
sns.scatterplot(x='whole_weight', y='shucked_weight', hue='type', data=abalone_df, palette='viridis')
plt.title('Whole Weight vs Shucked Weight by Type')
plt.xlabel('Whole Weight')
plt.ylabel('Shucked Weight')
plt.legend(title='Type')
plt.savefig('whole_weight_vs_shucked_weight_by_type.png')
plt.close()

# 5. Scatter plot for whole weight vs shell weight colored by type
plt.figure(figsize=(10, 6))
sns.scatterplot(x='whole_weight', y='shell_weight', hue='type', data=abalone_df, palette='viridis')
plt.title('Whole Weight vs Shell Weight by Type')
plt.xlabel('Whole Weight')
plt.ylabel('Shell Weight')
plt.legend(title='Type')
plt.savefig('whole_weight_vs_shell_weight_by_type.png')
plt.close()