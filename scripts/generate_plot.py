import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# --- Step 1: Simulate the input data ---
# In your actual project, you would load the 'spec_data' DataFrame that
# your 'data_labeler.py' script generates.
# For this example, we create plausible sample data.
print("Generating sample data for plotting...")
np.random.seed(42)
scores_low = np.random.normal(loc=30, scale=10, size=50)
scores_mid = np.random.normal(loc=80, scale=15, size=80)
scores_high = np.random.normal(loc=150, scale=20, size=40)
performance_scores = np.concatenate([scores_low, scores_mid, scores_high])
# Ensure scores are not negative
performance_scores[performance_scores < 0] = 5

sample_df = pd.DataFrame({'Performance_Score': performance_scores})


# --- Step 2: Run K-Means to get cluster labels and centers ---
# This step mirrors the logic in your data_labeler.py to create the tiers.
print("Running K-Means to assign cluster labels...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
sample_df['Cluster_Label'] = kmeans.fit_predict(sample_df[['Performance_Score']])
centroids = kmeans.cluster_centers_

# Map numeric labels to meaningful names based on centroid order
centroid_order = np.argsort(centroids.flatten())
label_mapping = {
    centroid_order[0]: 'Low-Tier',
    centroid_order[1]: 'Mid-Tier',
    centroid_order[2]: 'High-Tier'
}
sample_df['Hardware_Tier'] = sample_df['Cluster_Label'].map(label_mapping)


# --- Step 3: Generate the plot using Matplotlib and Seaborn ---
print("Generating the plot...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 4)) # Create a figure and an axes

# A stripplot is excellent for visualizing 1D cluster distributions.
# Jitter adds a small amount of random noise to the y-axis to prevent points
# from overlapping perfectly, making the density of each cluster visible.
sns.stripplot(
    x='Performance_Score',
    y='Hardware_Tier',
    data=sample_df,
    order=['Low-Tier', 'Mid-Tier', 'High-Tier'], # Ensure consistent y-axis order
    palette={'Low-Tier': 'skyblue', 'Mid-Tier': 'orange', 'High-Tier': 'lightgreen'},
    jitter=0.25,
    alpha=0.7,
    ax=ax
)

# Plot the cluster centroids as large red 'X's for emphasis.
# We get the y-positions (0, 1, 2) and map them to the ordered centroids.
y_coords = {'Low-Tier': 0, 'Mid-Tier': 1, 'High-Tier': 2}
ordered_centroids = centroids[centroid_order].flatten()

for tier_name, y_pos in y_coords.items():
    centroid_x = ordered_centroids[y_pos]
    ax.plot(centroid_x, y_pos, 'rX', markersize=12, label='Cluster Centroid' if y_pos == 0 else "")


# --- Step 4: Customize and save the plot ---
ax.set_title('K-Means Clustering of Devices by Performance Score', fontsize=16)
ax.set_xlabel('Calculated Performance Score', fontsize=12)
ax.set_ylabel('Assigned Hardware Tier', fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.legend()
plt.tight_layout() # Adjusts plot to ensure everything fits without overlapping

# Save the figure with the exact name your LaTeX document expects
output_filename = 'kmeans_placeholder.png'
plt.savefig(output_filename, dpi=300) # dpi=300 for high quality

print(f"\nSuccess! Plot saved as '{output_filename}'.")
print("You can now recompile your LaTeX document.")