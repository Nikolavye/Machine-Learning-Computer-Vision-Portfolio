import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Read data
df = pd.read_csv("../data_sensors.csv")
X = df.iloc[:, :20].values
labels = df["Label"].values

# Separate labeled and unlabeled data
labeled_mask = ~np.isnan(labels)
unlabeled_mask = np.isnan(labels)

# ============ PCA ============
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# ============ t-SNE ============
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

# ============ UMAP ============
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
X_umap = reducer.fit_transform(X)

# ============ Plotting ============
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

color_map = {1.0: "red", 2.0: "blue", 3.0: "green"}

for ax, X_proj, title in [
    (axes[0], X_pca, f"PCA (explained var: {pca.explained_variance_ratio_[:2].sum():.1%})"),
    (axes[1], X_tsne, "t-SNE (perplexity=30)"),
    (axes[2], X_umap, "UMAP (n_neighbors=15)"),
]:
    # Plot unlabeled points first (gray)
    ax.scatter(X_proj[unlabeled_mask, 0], X_proj[unlabeled_mask, 1],
               c="lightgray", s=10, alpha=0.5, label="Unlabeled")

    # Then plot labeled points (colored, larger)
    for label_val, color in color_map.items():
        mask = labels == label_val
        ax.scatter(X_proj[mask, 0], X_proj[mask, 1],
                   c=color, s=60, edgecolors="black", linewidth=0.5,
                   label=f"Label {int(label_val)}", zorder=5)

    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("Industrial Sensor Data Visualization (PCA vs t-SNE vs UMAP)", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig("visualization.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved: visualization.png")

# ============ PCA Extra Info ============
pca_full = PCA().fit(X)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
print(f"\nPCA explained variance (cumulative):")
for i, v in enumerate(cumvar):
    marker = " <-- 90%" if v >= 0.9 and (i == 0 or cumvar[i-1] < 0.9) else ""
    print(f"  PC{i+1:2d}: {v:.1%}{marker}")
