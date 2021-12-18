import pandas as pd
from classes import PNN

pnn = PNN()
pnn.load_data(trial=True)

results = pd.DataFrame(columns=['configuration', 'significance', 'train_last', 'train_best', 'val_last', 'val_best'])
# lrs = [0.1, 0.01, 0.001]
# hidden_sizes = [20, 80, 160]
# hidden_layers = [1, 2, 3]
# gammas = [0.9, 0.99, 0.999]
lrs = [0.1, 0.01]
hidden_sizes = [20, 80]
hidden_layers = [1]
gammas = [0.9]
n_trials = len(lrs) * len(hidden_sizes) * len(hidden_layers) * len(gammas)
n = 1

for lr in lrs:
    for hidden_size in hidden_sizes:
        for hidden_layer in hidden_layers:
            for gamma in gammas:
                print(f'Trial {n}/{n_trials}')
                train_history, val_history, sigma = pnn.train_validate(hidden_size, hidden_layer, 'sgd', 2, lr,
                                                                       500, 'relu', gamma)
                train_last = train_history[-1]
                val_last = val_history[-1]
                train_history.sort(key=lambda tupl: tupl[1])
                val_history.sort(key=lambda tupl: tupl[1])
                train_best = train_history[0]
                val_best = val_history[0]

                results = results.append({'configuration': f'lr = {lr}, hidden_size = {hidden_size},'
                                                           f' hidden_layers = {hidden_layer}, gamma = {gamma}',
                                          'significance': sigma, 'train_last': train_last, 'train_best': train_best,
                                          'val_last': val_last, 'val_best': val_best}, ignore_index=True)
                n += 1

