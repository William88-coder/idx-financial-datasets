from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

INPUT_FILE = BASE_DIR / "clustering_ready_2023_2024.csv"
OUTPUT_DIR = BASE_DIR / "cluster_charts"
RESULT_FILE = OUTPUT_DIR / "cluster_result_2023_2024.csv"

ID_COLUMNS = ["No", "Ticker", "Sector", "Sharia"]
EXCLUDED_PREFIX = "USD_IDR_"

K_MIN = 2
K_MAX = 10
MIN_AUTO_K = 3
N_CLUSTERS = None
N_INIT = 20
RANDOM_STATE = 42
SKEW_THRESHOLD = 1.0
LABEL_TOP_N = 8
LABEL_PLACEMENTS = [
    ((6, 6), "left"),
    ((6, -13), "left"),
    ((-6, 6), "right"),
    ((-6, -13), "right"),
]

BLUE = "#2563eb"
ORANGE = "#ea580c"
GREEN = "#16a34a"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def signed_log(values):
    return np.sign(values) * np.log1p(np.abs(values))


def load_dataset():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    feature_columns = [
        c for c in df.columns
        if c not in ID_COLUMNS
        and not c.startswith(EXCLUDED_PREFIX)
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[feature_columns].astype(float).replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median())
    X = X.loc[:, X.nunique() > 1]

    return df, X


def scale_features(X):
    skewness = X.skew().abs()
    skewed_columns = skewness[skewness > SKEW_THRESHOLD].index.tolist()

    transformed = X.copy()
    if skewed_columns:
        transformed[skewed_columns] = signed_log(transformed[skewed_columns])

    scaled = StandardScaler().fit_transform(transformed)
    return pd.DataFrame(scaled, columns=X.columns, index=X.index)


def evaluate_k(scaled):
    k_values = list(range(K_MIN, K_MAX + 1))
    inertias = []
    silhouettes = []

    for k in k_values:
        model = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE)
        labels = model.fit_predict(scaled)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(scaled, labels))

    return k_values, inertias, silhouettes


def choose_k(k_values, silhouettes):
    if N_CLUSTERS is not None:
        return int(N_CLUSTERS)

    eligible = [
        (score, k)
        for k, score in zip(k_values, silhouettes)
        if k >= MIN_AUTO_K
    ]
    return max(eligible)[1]


def fit_clusters(scaled, k):
    model = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE)
    raw_labels = pd.Series(model.fit_predict(scaled))

    size_order = raw_labels.value_counts().index
    mapping = {old: new for new, old in enumerate(size_order, start=1)}

    return raw_labels.map(mapping).to_numpy()


def cluster_colors(cluster_ids):
    palette = plt.get_cmap("tab10").colors
    return {c: palette[(c - 1) % len(palette)] for c in cluster_ids}


def plot_k_selection(k_values, inertias, silhouettes, best_k):
    fig, ax_left = plt.subplots(figsize=(10, 6))
    ax_right = ax_left.twinx()
    ax_right.grid(False)
    ax_right.spines["right"].set_visible(True)
    ax_left.margins(x=0.04, y=0.14)
    ax_right.margins(y=0.14)

    line_inertia, = ax_left.plot(
        k_values, inertias,
        marker="o", color=BLUE, linewidth=2.2,
        label="Inertia (Elbow)",
    )
    line_silhouette, = ax_right.plot(
        k_values, silhouettes,
        marker="s", color=ORANGE, linewidth=2.2,
        label="Silhouette score",
    )
    line_best = ax_left.axvline(
        best_k, color=GREEN, linestyle="--", linewidth=1.8,
        label=f"K terpilih = {best_k}",
    )

    ax_left.set_xticks(k_values)
    ax_left.set_xlabel("Jumlah cluster (K)")
    ax_left.set_ylabel("Inertia", color=BLUE)
    ax_right.set_ylabel("Silhouette score", color=ORANGE)
    ax_left.tick_params(axis="y", colors=BLUE)
    ax_right.tick_params(axis="y", colors=ORANGE)
    ax_left.set_title("Penentuan Jumlah Cluster: Elbow dan Silhouette")

    ax_right.legend(
        handles=[line_inertia, line_silhouette, line_best],
        loc="upper right",
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout()
    return fig


def plot_pca_scatter(scaled, df, colors):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(scaled)

    points = pd.DataFrame(coords, columns=["PC1", "PC2"], index=df.index)
    points["Cluster"] = df["Cluster"]

    centroids = points.groupby("Cluster")[["PC1", "PC2"]].mean()
    sizes = df["Cluster"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(11, 8))

    for cluster, group in points.groupby("Cluster"):
        ax.scatter(
            group["PC1"], group["PC2"],
            s=34, alpha=0.7, color=colors[cluster],
            edgecolor="white", linewidth=0.4,
            label=f"Cluster {cluster} (n={sizes[cluster]})",
        )

    for cluster, row in centroids.iterrows():
        ax.scatter(
            row["PC1"], row["PC2"],
            marker="X", s=300, color=colors[cluster],
            edgecolor="black", linewidth=1.6, zorder=5,
        )

    distance = np.hypot(points["PC1"], points["PC2"]).sort_values(ascending=False)
    for rank, idx in enumerate(distance.head(LABEL_TOP_N).index):
        offset, align = LABEL_PLACEMENTS[rank % len(LABEL_PLACEMENTS)]
        ax.annotate(
            df.loc[idx, "Ticker"],
            (points.loc[idx, "PC1"], points.loc[idx, "PC2"]),
            xytext=offset, textcoords="offset points", ha=align,
            fontsize=8.5, color="#333333",
        )

    ax.margins(0.06)

    ratio = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ratio[0]:.1%} varians)")
    ax.set_ylabel(f"PC2 ({ratio[1]:.1%} varians)")
    ax.set_title("Sebaran Cluster Saham (PCA 2D)")
    ax.legend(loc="best", frameon=True, framealpha=0.95)

    fig.tight_layout()
    return fig


def plot_cluster_profile(scaled, df, colors):
    profile = scaled.groupby(df["Cluster"]).mean().T
    sizes = df["Cluster"].value_counts().sort_index()

    matrix = profile.to_numpy()
    limit = min(float(np.abs(matrix).max()), 2.5)

    n_features, n_clusters = matrix.shape
    fig, ax = plt.subplots(
        figsize=(max(9, 1.3 * n_clusters + 5), max(8, 0.36 * n_features + 2))
    )

    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.grid(False)

    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels([f"Cluster {c}\n(n={sizes[c]})" for c in profile.columns])
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(profile.index)
    ax.tick_params(length=0)

    for tick, cluster in zip(ax.get_xticklabels(), profile.columns):
        tick.set_color(colors[cluster])
        tick.set_fontweight("bold")

    for i in range(n_features):
        for j in range(n_clusters):
            value = matrix[i, j]
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center", fontsize=8.5,
                color="white" if abs(value) > 0.6 * limit else "black",
            )

    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    colorbar.set_label("Rata-rata nilai terstandarisasi (z-score)")
    colorbar.outline.set_visible(False)

    ax.set_title("Profil Cluster: Rata-rata Fitur per Cluster")

    fig.tight_layout()
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, X = load_dataset()
    scaled = scale_features(X)

    k_values, inertias, silhouettes = evaluate_k(scaled)
    best_k = choose_k(k_values, silhouettes)

    df["Cluster"] = fit_clusters(scaled, best_k)
    colors = cluster_colors(sorted(df["Cluster"].unique()))

    figures = {
        "chart_1_elbow_silhouette.png": plot_k_selection(
            k_values, inertias, silhouettes, best_k
        ),
        "chart_2_pca_scatter.png": plot_pca_scatter(scaled, df, colors),
        "chart_3_profil_cluster.png": plot_cluster_profile(scaled, df, colors),
    }

    for name, fig in figures.items():
        fig.savefig(OUTPUT_DIR / name, dpi=200, bbox_inches="tight")

    result_columns = [c for c in ID_COLUMNS if c in df.columns] + ["Cluster"]
    df[result_columns].to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")

    print(f"Ticker dianalisis : {len(df):,}")
    print(f"Fitur dipakai     : {scaled.shape[1]}")
    print(f"K terpilih        : {best_k}")
    print(f"Silhouette score  : {silhouette_score(scaled, df['Cluster']):.4f}")
    print()
    print("Jumlah ticker per cluster:")
    print(df["Cluster"].value_counts().sort_index().to_string())
    print()
    print(f"Output            : {OUTPUT_DIR}")

    plt.show()


if __name__ == "__main__":
    main()
