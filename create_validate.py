from prepare_data import make_dataset
from classes import PNN
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt

# Initialise PNN class
pnn = PNN()

# Storage
results = pd.DataFrame(columns=['configuration', 'significance', 'train_last', 'val_last', 'last_epoch', 'train_best',
                                'tb_epoch', 'val_best', 'vb_epoch', 'early_stop'], index=None)
history_dict = {}
significance_curves = []
n = 10
signal_masses = [300, 420, 440, 460, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 2000]
filepath = '..//output_temp/'    # PC
# filepath = './output_temp/'        # Server

trials = [0.0001, 0.001, 0.01, 0.1, 1, 10]
for scaling in trials:
    # Configure dataset
    print(f'Making dataset {scaling}...')
    make_dataset(f'{filepath}data_dict_{scaling}.pkl', signal_masses, blind_mass=1200,
                 scale_norm=scaling, validation=True, test=True,
                 signal_normalisation=False)

    # load dataset for training
    pnn.load_data(f'{filepath}data_dict_{scaling}.pkl', trial=True, validating=True)

    # Train model
    print(f'Training...')
    output = pnn.train_validate(hidden_size=50, hidden_layers=2, activation_function='relu', n_epochs=50, lr=1.4,
                                gamma=0.999, momentum=0.7, early_stopping=10, reset_limit=15)
    print('Training complete.')

    train_history = output[0]
    val_history = output[1]
    sigma = output[2]
    early_stop = output[3]
    best_state = output[4]
    final_state = output[5]

    # Save model output
    history_dict[f'{scaling}'] = [train_history, val_history]
    train_last = train_history[-1]
    val_last = val_history[-1]
    train_best_epoch = train_history.index(np.min(train_history))
    val_best_epoch = val_history.index(np.min(val_history))
    train_best = train_history[train_best_epoch]
    val_best = val_history[val_best_epoch]

    results = results.append({'configuration': f'lr = {1.4}, h_s = {50},'
                                               f' h_l = {2}, gamma = {0.999}, activation = relu',
                              'significance': sigma, 'train_last': train_last, 'val_last': val_last,
                              'last_epoch': len(train_history), 'train_best': train_best,
                              'tb_epoch': train_best_epoch + 1, 'val_best': val_best,
                              'vb_epoch': val_best_epoch + 1, 'early_stop': early_stop}, ignore_index=True)
    results.to_csv(f'{filepath}results.csv')
    with open(f'{filepath}history_dict.pkl', 'wb') as f:
        pickle.dump(history_dict, f)

    torch.save(best_state['state'], f'{filepath}best_state_{scaling}.pth')
    torch.save(final_state['state'], f'{filepath}final_state_{scaling}.pth')

    # Plot model output
    sig_plot = pnn.validate(f'{filepath}best_state_{scaling}.pth', 50, 2, f'{filepath}output_plot_{scaling}.png')
    significance_curves.append(sig_plot)
    fig = plt.figure()
    plt.plot(history_dict[f'{scaling}'][0], label='signal')
    plt.plot(history_dict[f'{scaling}'][1], label='background')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(f'{filepath}loss_curves_{scaling}.png')
    n += 1
    print('Model output saved.')

plt.figure()
for curve, scaling in zip(significance_curves, trials):
    plt.plot(signal_masses, curve, label=scaling)

plt.xlabel('signal mass')
plt.ylabel('significance')
plt.legend(title='signal weight scaling')
plt.savefig(f'{filepath}comparison_plot.png')
