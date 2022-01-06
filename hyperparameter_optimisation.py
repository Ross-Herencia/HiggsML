import pandas as pd
import torch
from classes import PNN

pnn = PNN()
pnn.load_data(trial=True)

results = pd.DataFrame(columns=['configuration', 'significance', 'train_last', 'val_last', 'last_epoch', 'train_best',
                                'tb_epoch', 'val_best', 'vb_epoch', 'early_stop'], index=None)

lrs = [0.5, 1, 1.5, 2.0]
hidden_sizes = [20, 80, 160]
hidden_layers = [2, 3, 4]
gammas = [0.99, 0.999, 0.9999]
n_trials = len(lrs) * len(hidden_sizes) * len(hidden_layers) * len(gammas)
n = 1
val_best_state = {'state': None, 'val_best': 99, 'trial': n}
sigma_best_state = {'state': None, 'sigma': 0, 'trial': n}


for lr in lrs:
    for hidden_size in hidden_sizes:
        for hidden_layer in hidden_layers:
            for gamma in gammas:
                print(f'Trial {n}/{n_trials}')
                train_history, val_history, sigma, early_stop, best_state, final_state = pnn.train_validate(hidden_size, hidden_layer, 5,
                                                                                                lr, gamma, early_stopping=3, reset_limit=1)

                train_last = train_history[-1]
                val_last = val_history[-1]
                train_history.sort(key=lambda tupl: tupl[1])
                val_history.sort(key=lambda tupl: tupl[1])
                train_best = train_history[0]
                val_best = val_history[0]

                results = results.append({'configuration': f'lr = {lr}, h_s = {hidden_size},'
                                                           f' h_l = {hidden_layer}, gamma = {gamma}',
                                          'significance': sigma, 'train_last': train_last[1], 'val_last': val_last[1],
                                          'last_epoch': train_last[0], 'train_best': train_best[1],
                                          'tb_epoch': train_best[0], 'val_best': val_best[1], 'vb_epoch': val_best[0],
                                          'early_stop': early_stop}, ignore_index=True)
                
                if best_state['val_best'] < val_best_state['val_best']:
                    val_best_state = best_state
                    val_best_state['trial'] = n

                if final_state['sigma'] > sigma_best_state['sigma']:
                    sigma_best_state = final_state
                    sigma_best_state['trial'] = n

                n += 1
                results.to_csv('results.csv')


n = val_best_state['trial']  # Save trial number to cross-check with results that the correct state has been saved
torch.save(val_best_state['state'], f'vale_best_state_{n}.pth')
n = sigma_best_state['trial']
torch.save(sigma_best_state['state'], f'sigma_best_state_{n}.pth')