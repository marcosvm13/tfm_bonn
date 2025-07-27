import cupy as np
import seaborn as sns
import matplotlib.pyplot as plt


def model_training(pc, s, num, mode, variable1="s", variable2="pc", variable3="hidden_layer_mult", epochs=40, lr=0.1, cte=1, n_train=1):
    """Entrena un modelo con parámetros definidos y devuelve precisión por época."""
    accs = plotFrame(
        num if mode != "perceptron" else 25 / (15 * 15),
        [s], [pc], variable1, variable2, variable3,
        epochs, mode,
        lr if mode != "perceptron" else 0.01,
        cte, trys=1, n_train=n_train, registros=[1], verbose=True
    )
    return accs


def n_for_model(models_mode, s, pc, num, n_train, epochs=40, lr=0.1, cte=1, tries=5):
    """Evalúa precisión de múltiples modelos con diferentes tamaños de entrenamiento."""
    accs = {}
    for i in range(len(n_train)):
        accs[n_train[i]] = {}
        for mode, s_, pc_ in zip(models_mode, s, pc):
            key = f"{mode}_{s_}_{pc_}"
            accs[n_train[i]][key] = []
            for _ in range(tries):
                acc = model_training(
                    pc_, s_, num, mode, epochs=epochs, lr=lr, cte=cte, n_train=n_train[i])
                accs[n_train[i]][key].append(list(acc.values())[0][0][0][0])
    return accs


def plot_model_accuracies(data, epochs):
    """Gráfica de precisión media con sombreado de desviación estándar por modelo."""
    training_lengths = sorted(data.keys())
    models = list(next(iter(data.values())).keys())
    real_model_names_dict = {"OW": "Random Feature Model", "ALL": "MBBO",
                             "OB": "Only Bias Learning", "perceptron": "Perceptron"}
    models_real = [
        real_model_names_dict.get(model.split("_")[0], model.split(
            "_")[0]) + f" (pc={model.split('_')[1]} - s={model.split('_')[2]})"
        if model.split("_")[0] != "perceptron" else real_model_names_dict[model.split("_")[0]]
        for model in models
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    for model, model_name in zip(models, models_real):
        accuracies = [np.mean(data[length][model])
                      for length in training_lengths]
        std_devs = [np.sqrt(np.var(data[length][model]))
                    for length in training_lengths]
        ax.plot(training_lengths, accuracies,
                label=model_name, marker='o', linewidth=2.5)
        ax.fill_between(training_lengths, np.array(accuracies) -
                        std_devs, np.array(accuracies) + std_devs, alpha=0.2)

    ax.set_xlabel('Training Set Length', fontsize=16, weight='bold')
    ax.set_ylabel('Accuracy', fontsize=16, weight='bold')
    ax.set_title(
        f'Model Accuracies ({epochs} Epochs)', fontsize=18, weight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('model_accuracies.png', dpi=300)
    plt.show()


def plot_model_accuracies_2(data, epochs):
    """Versión estilizada para publicaciones de la función anterior."""
    sns.set(style="whitegrid")
    training_lengths = sorted(data.keys())
    models = list(next(iter(data.values())).keys())
    real_model_names_dict = {"OW": "Random Feature Model", "ALL": "MBBO",
                             "OB": "Only Bias Learning", "perceptron": "Perceptron"}
    models_real = [
        f"{real_model_names_dict.get(model.split('_')[0], model.split('_')[0])} (pc= {model.split('_')[1]} - s={model.split('_')[2]})"
        if model.split("_")[0] != "perceptron" else real_model_names_dict[model.split("_")[0]]
        for model in models
    ]

    colors = sns.color_palette("tab10", len(models))
    fig, ax = plt.subplots(figsize=(10, 8))

    for idx, (model, model_name) in enumerate(zip(models, models_real)):
        accuracies = [np.mean(data[length][model])
                      for length in training_lengths]
        std_devs = [np.sqrt(np.var(data[length][model]))
                    for length in training_lengths]
        ax.plot(training_lengths, accuracies, label=model_name,
                color=colors[idx], marker='o', linewidth=2.5)
        ax.fill_between(training_lengths, np.array(accuracies) - std_devs,
                        np.array(accuracies) + std_devs, color=colors[idx], alpha=0.3)

    ax.set_xlabel('Training Set Length', fontsize=12, color="#011f4b")
    ax.set_ylabel('Accuracy', fontsize=12, color="#011f4b")
    ax.set_ylim(0, 1)
    ax.grid(which="major", axis='x', color='#DAD8D7', alpha=0.5)
    ax.grid(which="major", axis='y', color='#DAD8D7', alpha=0.5)
    ax.legend(title='Models', fontsize=12,
              title_fontsize='13', loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig('model_accuracies_publication.svg', bbox_inches='tight')
    plt.show()
