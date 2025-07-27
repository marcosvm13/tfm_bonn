import matplotlib.pyplot as plt
import seaborn as sns
import numpy as npi


def plot_pruning_progress_stylized(percent_deleted, accuracy, title="Neuron Pruning Impact on Accuracy", percent=False):
    """
    Grafica la evolución de la precisión frente a la proporción o número de neuronas eliminadas.

    Parámetros:
    - percent_deleted: lista de listas con los porcentajes o números de neuronas eliminadas por modelo.
    - accuracy: lista de listas con las precisiones correspondientes a cada nivel de eliminación.
    - title: título del gráfico.
    - percent: si True, el eje X representa porcentajes; si False, número de neuronas.
    """
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.4)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e']
    model = ["BONN", "MLP"]

    for i, (acc, p) in enumerate(zip(accuracy, percent_deleted)):
        ax.plot(p, acc, marker='o',
                color=colors[i], linewidth=2, markersize=4, label=model[i])

    ax.set_ylim(0, 1.05)
    ax.set_yticks(npi.linspace(0, 1, 11))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))

    if percent:
        ax.set_xticks(npi.linspace(0, 100, 11))
        ax.set_xlabel("Neurons Pruned (%)", fontsize=14,
                      weight='bold', labelpad=10)
    else:
        ax.set_xticks(npi.linspace(
            0, max([max(per) for per in percent_deleted]), 11))
        ax.set_xlabel("Number of Dead Neurons", fontsize=14,
                      weight='bold', labelpad=10)

    ax.set_ylabel("Accuracy", fontsize=14, weight='bold', labelpad=10)
    ax.grid(color="#DDDDDD", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis='both', labelsize=12, colors="#011f4b", width=1.2)

    legend = ax.legend(
        title="Models",
        loc='upper right',
        ncol=2,
        fontsize=12,
        title_fontsize=13,
        frameon=True,
        fancybox=True,
    )
    legend.get_frame().set_facecolor('#f7f7f7')
    legend.get_frame().set_edgecolor('#cccccc')
    legend.get_frame().set_linewidth(0.8)

    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.85)
    plt.title(title, fontsize=16, weight='bold')
    plt.savefig("pruning.eps", format="eps", bbox_inches='tight')
    plt.show()
