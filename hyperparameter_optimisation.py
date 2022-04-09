import pandas as pd
import torch
from classes import PNN
import pickle
import numpy as np

pnn = PNN()
pnn.load_data(trial=True)

results = pd.DataFrame(columns=['configuration', 'significance', 'train_last', 'val_last', 'last_epoch', 'train_best',
                                'tb_epoch', 'val_best', 'vb_epoch', 'early_stop'], index=None)
history_dict = {}

lrs = [1.4]
hidden_sizes = [50]
hidden_layers = [2]
gammas = [0.999]
activation_functions = ['relu']
momentum = 0.7


n_trials = len(lrs) * len(hidden_sizes) * len(hidden_layers) * len(gammas) * len(activation_functions)
n = 0
val_best_state = {'state': None, 'val_best': 99, 'trial': 999}
sigma_best_state = {'state': None, 'sigma': 0, 'trial': 999}


for lr in lrs:
    for hidden_size in hidden_sizes:
        for hidden_layer in hidden_layers:
            for gamma in gammas:
                for activation in activation_functions:

                    print(f'Trial {n + 1}/{n_trials}')

                    train_history, val_history, sigma, early_stop, temp_best_state, temp_final_state = pnn.train_validate(hidden_size, hidden_layer, activation, 100,
                                                                                                    lr, gamma, momentum, early_stopping=10, reset_limit=15)

                    history_dict[f'{n}'] = [train_history, val_history]
                    train_last = train_history[-1]
                    val_last = val_history[-1]
                    train_best_epoch = train_history.index(np.min(train_history))
                    val_best_epoch = val_history.index(np.min(val_history))
                    train_best = train_history[train_best_epoch]
                    val_best = val_history[val_best_epoch]

                    results = results.append({'configuration': f'lr = {lr}, h_s = {hidden_size},'
                                                f' h_l = {hidden_layer}, gamma = {gamma}, activation = {activation}',
                                                'significance': sigma, 'train_last': train_last, 'val_last': val_last,
                                                'last_epoch': len(train_history), 'train_best': train_best,
                                                'tb_epoch': train_best_epoch + 1, 'val_best': val_best,
                                                'vb_epoch': val_best_epoch + 1, 'early_stop': early_stop}, ignore_index=True)

                    if temp_best_state['val_best'] < val_best_state['val_best']:
                        val_best_state = temp_best_state
                        val_best_state['trial'] = n  # to match pandas index

                    if temp_final_state['sigma'] > sigma_best_state['sigma']:
                        sigma_best_state = temp_final_state
                        sigma_best_state['trial'] = n

                    n += 1
                    results.to_csv('00-results.csv')
                    with open('00-history_dict.pkl', 'wb') as f:
                        pickle.dump(history_dict, f)


n = val_best_state['trial']  # Save trial number to cross-check with results that the correct state has been saved
torch.save(val_best_state['state'], f'00-val_best_state_{n}.pth')
n = sigma_best_state['trial']
torch.save(sigma_best_state['state'], f'00-sigma_best_state_{n}.pth')
