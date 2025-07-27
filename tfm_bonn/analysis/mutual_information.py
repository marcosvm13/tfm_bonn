import cupy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score


def compute_mi_matrix(kc_activations, labels, num_kc: int = 10):
    """
    Calcula la matriz de información mutua entre activaciones de KCs y etiquetas.

    Args:
        kc_activations (np.ndarray): Activaciones binarias de KCs (N muestras x D neuronas).
        labels (np.ndarray): Etiquetas de clase por muestra.
        num_kc (int): Número de etiquetas (clases).

    Returns:
        np.ndarray: Matriz de información mutua (num_kc x num_kc).
    """
    mi_matrix = np.zeros((num_kc, num_kc))

    for i in range(num_kc):
        for j in range(i, num_kc):
            idx_i = np.where(labels == i)[0]
            idx_j = np.where(labels == j)[0]
            a = min(len(idx_i), len(idx_j))
            x = kc_activations[idx_i][:a]
            y = kc_activations[idx_j][:a]
            mi_value_mean = 0
            for xi, yi in zip(x, y):
                c_xy = np.histogram2d(xi, yi, 10)[0]
                mi_value_mean += mutual_info_score(None,
                                                   None, contingency=c_xy)
            mi_matrix[i, j] = mi_matrix[j, i] = mi_value_mean / len(x)

    return mi_matrix


def plot_two_silhouettes(silhouette_matrix_1, model_name_1,
                         silhouette_matrix_2, model_name_2,
                         unique_labels, pc=None, s=None, acc1=None, acc2=None):
    """
    Dibuja dos mapas de calor con matrices de silueta para comparar modelos.
    """
    max_val = max(np.max(silhouette_matrix_1), np.max(silhouette_matrix_2))
    scale_exp = int(np.floor(np.log10(max_val))) if max_val > 0 else 0
    scale_factor = 10 ** scale_exp

    scaled_1 = silhouette_matrix_1 / scale_factor
    scaled_2 = silhouette_matrix_2 / scale_factor

    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.4)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)

    cmap = 'coolwarm'
    vmin = min(np.min(silhouette_matrix_1), np.min(silhouette_matrix_2))
    vmax = max(np.max(silhouette_matrix_1), np.max(silhouette_matrix_2))

    im1 = sns.heatmap(silhouette_matrix_1, annot=scaled_1, fmt=".3f", cmap=cmap,
                      xticklabels=unique_labels, yticklabels=unique_labels,
                      ax=axes[0], cbar=False, linewidths=0.5, linecolor='grey',
                      square=True, annot_kws={"size": 11}, vmin=vmin, vmax=vmax)

    im2 = sns.heatmap(silhouette_matrix_2, annot=scaled_2, fmt=".3f", cmap=cmap,
                      xticklabels=unique_labels, yticklabels=unique_labels,
                      ax=axes[1], cbar=False, linewidths=0.5, linecolor='grey',
                      square=True, annot_kws={"size": 11}, vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(im2.collections[0], ax=axes, orientation='vertical',
                        fraction=0.025, pad=0.03)
    cbar.set_label(
        f'Mutual Information ($\\times 10^{{{scale_exp}}}$)', fontsize=14, weight='bold', labelpad=10)

    axes[0].set_title(f"{model_name_1}", fontsize=16, weight='bold', pad=10)
    axes[1].set_title(f"{model_name_2}", fontsize=16, weight='bold', pad=10)

    for ax in axes:
        ax.set_xlabel("Class", fontsize=14, weight='bold')
        ax.set_ylabel("Class", fontsize=14, weight='bold')
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, which="major", axis='both', color='#DAD8D7', alpha=0.5)
        ax.spines[['top', 'right']].set_visible(False)

    plt.savefig(f"Silhouettes_{model_name_1}_vs_{model_name_2}.eps",
                format="eps", bbox_inches='tight')
    plt.show()
