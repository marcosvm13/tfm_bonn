import cupy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
import numpy as npi


def plot_surfaces_adjusted(accs_list, accs_var, primera_variable, segunda_variable, tercera_variable, perceptron_max_acc, model_name):
    """
    Dibuja 4 superficies de precisión con una barra de color compartida.
    """
    sns.set(style="whitegrid")
    pc_values = npi.array(segunda_variable)
    s_values = npi.array(primera_variable)
    pc_grid, s_grid = npi.meshgrid(pc_values, s_values)

    fig, axes = plt.subplots(2, 2, figsize=(
        14, 12), constrained_layout=False, sharex=True, sharey=True)
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.1,
                        right=0.88, hspace=0.15, wspace=0.15)
    epoch_labels = ["1 Epoch", "2 Epochs", "5 Epochs", "10 Epochs"]
    vmin, vmax = 0, 1
    axes = axes.flatten()

    for i, (acc_matrix, var, ax, epoch_label, per_acc) in enumerate(zip(accs_list, accs_var, axes, epoch_labels, perceptron_max_acc.values())):
        acc_matrix = npi.array(acc_matrix)
        contour = ax.contourf(pc_grid, s_grid, acc_matrix,
                              cmap='viridis', levels=20, vmin=vmin, vmax=vmax)

        max_idx = npi.unravel_index(npi.argmax(acc_matrix), acc_matrix.shape)
        max_pc, max_s, max_acc, max_var = pc_values[max_idx[1]], s_values[max_idx[0]
                                                                          ], acc_matrix[max_idx[0]][max_idx[1]], var[max_idx[0]][max_idx[1]]

        ax.scatter(max_pc, max_s, color='red', s=100,
                   edgecolors='black', label=f'Max Accuracy: {max_acc:.2f}')
        textstr = (
            r"$\bf{Max \,\,\, Accuracy}$" "\n"
            r"MLP: " f"{per_acc[0]:.2f} ± {per_acc[1]:.2f}" "\n"
            r"BONN: " f"{max_acc:.2f} ± {max_var:.2f}"
        )
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', linewidth=0.8, alpha=1.0))
        ax.text(0.95, 0.95, epoch_label, transform=ax.transAxes, fontsize=16, weight='bold', verticalalignment='top',
                horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='black', alpha=0.85))
        ax.set_xticks(pc_values)
        ax.set_yticks(s_values)

    cbar = fig.colorbar(
        contour, ax=axes, orientation='vertical', fraction=0.02, pad=0.03)
    cbar.ax.set_ylabel('Accuracy', fontsize=18,
                       weight='bold', rotation=270, labelpad=30)
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.tick_params(labelsize=14, pad=10)
    fig.text(0.5, 0.08, 'Connectivity Parameter (pc)', ha='center',
             fontsize=16, weight='bold', color='#6497b1')
    fig.text(0.02, 0.5, 'Activity Parameter (s)', va='center',
             rotation='vertical', fontsize=16, weight='bold', color='#6497b1')
    plt.savefig("medias.eps", format='eps', dpi=300)
    plt.show()
