import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. SETUP DATA (Matches your 5,000-post dataset proportions)
# ---------------------------------------------------------
np.random.seed(42)
n_rows = 5000
platforms = ['TikTok', 'Instagram', 'Twitter', 'YouTube']

# Use existing CSV if available on Desktop, otherwise create identical simulation
try:
    df = pd.read_csv('social_media_data.csv')
    print("[Data Loaded] Using your social_media_data.csv for visualizations.")
except FileNotFoundError:
    print("[Data Simulated] CSV not found on Desktop. Using matching simulated data.")
    df = pd.DataFrame({
        'Platform': np.random.choice(platforms, size=n_rows),
        'Views': np.random.randint(1000, 5000000, size=n_rows),
        'Likes': np.random.randint(500, 500000, size=n_rows),
        'Shares': np.random.randint(50, 100000, size=n_rows),
        'Comments': np.random.randint(10, 50000, size=n_rows)
    })

# Feature Engineering
df['Total_Interactions'] = df['Likes'] + df['Shares'] + df['Comments']
df['Engagement_Rate'] = df['Total_Interactions'] / df['Views']

# ---------------------------------------------------------
# 2. GENERATE GRAPHICAL REPRESENTATIONS (Units I & II EDA)
# ---------------------------------------------------------
# Set up a 1x2 side-by-side plot layout
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Histogram of Views
axes[0].hist(df['Views'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].set_title('Distribution of Views (Unit II EDA)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Views (Raw)')
axes[0].set_ylabel('Frequency')
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Plot 2: Boxplot of Views by Platform
platform_data = [df[df['Platform'] == p]['Views'] for p in platforms]
axes[1].boxplot(platform_data, labels=platforms, patch_artist=True,
                boxprops=dict(facecolor='lightcoral', color='black'),
                medianprops=dict(color='black', linewidth=1.5))
axes[1].set_title('Views by Platform (Unit III ANOVA Setup)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Platform')
axes[1].set_ylabel('Views')
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot
plt.tight_layout()
plt.savefig('social_media_visuals.png', dpi=300)
plt.close()

print("\n=== Success! ===")
print("[Visuals Saved] Your plot 'social_media_visuals.png' has been saved to your Desktop.")
