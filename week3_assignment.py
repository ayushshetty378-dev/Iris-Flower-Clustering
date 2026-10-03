import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, confusion_matrix

COLS = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", "species"]
candidates = ["iris.data", "iris.csv", "Iris.csv", "iris.data.csv"]
csv_path = next((Path(c) for c in candidates if Path(c).exists()), None)
if csv_path is None:
    raise FileNotFoundError("Put iris.data (or iris.csv) in the same folder as this script.")

with open(csv_path) as f:
    has_header = f.readline()[0].isalpha()          # UCI file has no header row
df = pd.read_csv(csv_path, header=0 if has_header else None)
df = df.drop(columns=[c for c in df.columns if str(c).lower() == "id"])  # Kaggle adds an Id column
df = df.dropna().iloc[:, -5:]                       # keep 4 features + species
df.columns = COLS
print(f"Loaded {csv_path} -> {df.shape[0]} rows, {df.shape[1]} columns")

features = COLS[:4]
X = df[features].to_numpy(dtype=float)
species = df["species"].str.replace("Iris-", "", regex=False)
names = sorted(species.unique())
y = species.map({n: i for i, n in enumerate(names)}).to_numpy()
X_scaled = StandardScaler().fit_transform(X)

ks = range(1, 11)
inertias = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled).inertia_ for k in ks]
plt.figure(figsize=(6, 4))
plt.plot(ks, inertias, "o-")
plt.axvline(3, ls="--", c="gray")
plt.xlabel("Number of clusters (K)"); plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method"); plt.tight_layout()
plt.savefig("elbow.png", dpi=150); plt.close()

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)

print("=== K-Means cluster centers (scaled space) ===")
print(pd.DataFrame(kmeans.cluster_centers_, columns=features).round(3))
print("\n=== K-Means cluster centers (original units, cm) ===")
centers_orig = pd.DataFrame(
    [X[labels == c].mean(axis=0) for c in range(3)], columns=features
).round(3)
print(centers_orig)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\n=== PCA explained variance ratio ===")
print("PC1: %.4f  PC2: %.4f  Total: %.4f" % (*pca.explained_variance_ratio_, pca.explained_variance_ratio_.sum()))
full_pca = PCA().fit(X_scaled)
print("All 4 components:", full_pca.explained_variance_ratio_.round(4))

centers_pca = pca.transform(kmeans.cluster_centers_)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, color_by, title in [(axes[0], labels, "K-Means Predicted Clusters (k=3)"),
                            (axes[1], y, "True Species Labels")]:
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=color_by, cmap="viridis", edgecolor="k", s=50)
    if color_by is labels:
        ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c="red", marker="X", s=250, edgecolor="k", label="Centroids")
        ax.legend()
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(title)
plt.tight_layout(); plt.savefig("clusters_vs_true.png", dpi=150); plt.close()

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 2], X[:, 3], c=labels, cmap="viridis", edgecolor="k", s=50)
plt.scatter(centers_orig.iloc[:, 2], centers_orig.iloc[:, 3], c="red", marker="X", s=250, edgecolor="k")
plt.xlabel(features[2]); plt.ylabel(features[3]); plt.title("K-Means Clusters: Petal Length vs Width")
plt.tight_layout(); plt.savefig("petal_scatter.png", dpi=150); plt.close()

print("\n=== Predicted clusters vs true labels ===")
print(pd.crosstab(species.values, labels,
                  rownames=["True species"], colnames=["Cluster"]))
print("\nAdjusted Rand Index: %.4f" % adjusted_rand_score(y, labels))

cm = confusion_matrix(y, labels)
mapping = {c: cm[:, c].argmax() for c in range(3)}
mapped = np.array([mapping[l] for l in labels])
print("Cluster-to-species accuracy: %.2f%%" % ((mapped == y).mean() * 100))

from sklearn.metrics import silhouette_score

evr = full_pca.explained_variance_ratio_
plt.figure(figsize=(6, 4))
plt.bar(range(1, 5), evr, alpha=0.7, label="Individual")
plt.plot(range(1, 5), np.cumsum(evr), "ro-", label="Cumulative")
plt.xticks(range(1, 5)); plt.xlabel("Principal component"); plt.ylabel("Explained variance ratio")
plt.title("PCA Explained Variance"); plt.legend(); plt.tight_layout()
plt.savefig("pca_variance.png", dpi=150); plt.close()

sil_ks = range(2, 11)
sil = [silhouette_score(X_scaled, KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_scaled)) for k in sil_ks]
plt.figure(figsize=(6, 4))
plt.plot(sil_ks, sil, "o-")
plt.xlabel("Number of clusters (K)"); plt.ylabel("Silhouette score")
plt.title("Silhouette Score vs K"); plt.tight_layout()
plt.savefig("silhouette.png", dpi=150); plt.close()

fig, axes = plt.subplots(4, 4, figsize=(11, 10))
for i in range(4):
    for j in range(4):
        ax = axes[i, j]
        if i == j:
            for c in range(3):
                ax.hist(X[labels == c, i], bins=12, alpha=0.6)
        else:
            ax.scatter(X[:, j], X[:, i], c=labels, cmap="viridis", s=12)
        if i == 3: ax.set_xlabel(features[j].replace(" (cm)", ""), fontsize=8)
        if j == 0: ax.set_ylabel(features[i].replace(" (cm)", ""), fontsize=8)
fig.suptitle("Pair Plot Coloured by K-Means Cluster", y=0.995)
plt.tight_layout(); plt.savefig("pairplot.png", dpi=130); plt.close()

order = [mapping[c] for c in range(3)]
cm_plot = confusion_matrix(y, mapped)
plt.figure(figsize=(5, 4))
plt.imshow(cm_plot, cmap="Blues")
for (r, c), v in np.ndenumerate(cm_plot):
    plt.text(c, r, v, ha="center", va="center", color="white" if v > cm_plot.max() / 2 else "black")
plt.xticks(range(3), names); plt.yticks(range(3), names)
plt.xlabel("Cluster (mapped to species)"); plt.ylabel("True species"); plt.title("Clusters vs True Labels")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()

plt.figure(figsize=(5, 4))
sizes = np.bincount(labels)
plt.bar([f"Cluster {i}" for i in range(3)], sizes, color=["#440154", "#21918c", "#fde725"], edgecolor="k")
for i, v in enumerate(sizes): plt.text(i, v + 1, str(v), ha="center")
plt.ylabel("Number of flowers"); plt.title("Cluster Sizes"); plt.tight_layout()
plt.savefig("cluster_sizes.png", dpi=150); plt.close()

print("\nExtra graphs saved: pca_variance, silhouette, pairplot, confusion_matrix, cluster_sizes")

joblib.dump(kmeans, "kmeans_iris.joblib")
loaded = joblib.load("kmeans_iris.joblib")
assert (loaded.predict(X_scaled) == labels).all()
print("\nModel saved to kmeans_iris.joblib and reloaded successfully.")
