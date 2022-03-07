import torch
import torch.nn as nn
import numpy as np
import pickle
from functions import significance
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


class NN(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layers, activation_function, output_size):
        super(NN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation_function = activation_function
        self.layers = nn.ModuleList()

        input = input_size
        for layer in range(self.hidden_layers):
            self.layers.append(nn.Linear(input, hidden_size))
            input = hidden_size
        self.layers.append(nn.Linear(input, output_size))

        activation_funcs = {'relu': nn.ReLU(), 'selu': nn.SELU(), 'lrelu': nn.LeakyReLU(), 'prelu': nn.PReLU()}
        self.activation = activation_funcs[activation_function]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        out = self.sigmoid(self.layers[-1](x))

        return out


class PNN:
    def __init__(self):
        self.train_data = []
        self.val_data = []
        self.validating = {}
        self.testing = {}

    def load_data(self, data_dict_filepath,trial=False, validating=False, testing=False):

        if (not trial) & (not validating) & (not testing):
            print('No data has been selected. Set at least one of trial, validating or testing = True.')
            exit(1)

        with open(data_dict_filepath, 'rb') as f:    # PC
        # with open('./data_dict.pkl', 'rb') as f:    # Server

            data = pickle.load(f)

        # data = {key: [x, y, weights], ...}

        keys = list(data.keys())

        if trial:
            self.train_data = data['train']
            self.val_data = data['val']

        if validating:
            self.val_data = data['val']
            self.validating = {key: data[key] for key in keys[3:16]}

        if testing:
            self.testing = {key: data[key] for key in keys[16:]}

        del data

    def train_validate(self, hidden_size, hidden_layers, activation_function, n_epochs, lr, gamma, momentum, early_stopping=10, reset_limit=10):

        # Default hyperparameters which will remain fixed
        input_size = 20
        output_size = 1
        optimisation_function = 'sgd'
        batch_size = 500
        # activation_function = 'relu'

        # Storage
        train_history = []
        val_history = []
        prob_weights_labels = np.zeros((len(self.val_data[0]), 3))
        count = 0
        reset_count = 0
        early_stop = 0
        val_best = 99.0

        # Model initialisation
        model = NN(input_size, hidden_size, hidden_layers, activation_function, output_size)
        optimiser_funcs = {'sgd': torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True),
                           'adam': torch.optim.Adam(model.parameters(), lr=lr)}
        optimiser = optimiser_funcs[optimisation_function]
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=gamma)

        # Training and validation loop
        for epoch in range(n_epochs):
            model.train()
            train_loss = 0
            self.train_data[0], self.train_data[1], self.train_data[2] = shuffle(self.train_data[0], self.train_data[1],
                                                                                 self.train_data[2],
                                                                                 random_state=np.random.randint(0, 100))
            for a in np.random.permutation(int(np.ceil(len(self.train_data[0]) / batch_size))):
                batch = self.train_data[0][a * batch_size: (a + 1) * batch_size, :]
                labels = self.train_data[1][a * batch_size: (a + 1) * batch_size]
                weights = self.train_data[2][a * batch_size: (a + 1) * batch_size]

                loss_fn = nn.BCELoss(torch.abs(weights.unsqueeze(1)))
                out = model(batch)
                loss = loss_fn(out, labels.unsqueeze(1))
                train_loss += loss.item()
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            train_history.append(train_loss / np.ceil(len(self.train_data[0]) / batch_size))

            model.eval()
            val_loss = 0
            with torch.no_grad():
                batch = self.val_data[0]
                labels = self.val_data[1]
                weights = self.val_data[2]

                loss_fn = nn.BCELoss(torch.abs(weights.unsqueeze(1)))
                out = model(batch)
                loss = loss_fn(out, labels.unsqueeze(1))
                val_loss += loss.item()

            val_history.append(val_loss)

            # Early stopping
            if (epoch > 0) & (val_history[epoch] >= val_history[epoch - 1]):
                count += 1
                if count == early_stopping:
                    early_stop = 1
                    break
            elif (count > 0) & (val_history[epoch] < val_best):
                count = 0   # reset counter if there is improvement again
                reset_count += 1
            if reset_count >= reset_limit:
                early_stop = 2  # oscillating validation loss
                break

            # Save best validation loss model
            if (epoch > 0) & (val_history[epoch] < val_best):
                val_best = val_history[-1]
                best_state = {'state': model.state_dict(), 'val_best': val_best}

            scheduler.step()

        prob_weights_labels[:, 0] = out[:, 0]
        prob_weights_labels[:, 1] = weights
        prob_weights_labels[:, 2] = labels
        s, b, sigma = significance(prob_weights_labels, 10)

        final_state = {'state': model.state_dict(), 'sigma': sigma}

        return [train_history, val_history, sigma, early_stop, best_state, final_state]

    def validate(self, model_path, hidden_size, hidden_layers, plot_filename):
        input_size = 20
        activation_function = 'relu'
        output_size = 1
        model = NN(input_size, hidden_size, hidden_layers, activation_function, output_size)
        model.load_state_dict(torch.load(model_path))

        significances = []
        significances_norms = []
        rows = 5
        cols = 3

        plt.rcParams.update({'font.size': 30})

        fig, ax = plt.subplots(rows, cols, figsize=(50, 50))
        i = 0
        j = 0
        model.eval()
        with torch.no_grad():
            for key, value in self.validating.items():
                prob_weights_labels = np.zeros((len(value[0]), 3))
                batch = value[0]
                labels = value[1]
                weights = value[2]

                out = model(batch)
                prob_weights_labels[:, 0] = out[:, 0]
                prob_weights_labels[:, 1] = weights
                prob_weights_labels[:, 2] = labels

                s, b, sigma = significance(prob_weights_labels, bins=20)
                significances.append(sigma)
                x = np.arange(0, 1, 1 / len(s))
                width = 1 / len(s)
                ax[i, j].bar(x, s, width=width, align='edge', label='signal', color='red', alpha=0.5)
                ax[i, j].bar(x, b, width=width, align='edge', label='background', color='blue', alpha=0.5)
                ax[i, j].set_title(f'{key}')
                ax[i, j].set_yscale('log')
                ax[i, j].legend()

                j += 1
                if j == cols:
                    j = 0
                    i += 1

                s, b, sigma = significance(prob_weights_labels, normalise=True)
                significances_norms.append(sigma)

            for axs in ax.flat:
                axs.set(xlabel='probability', ylabel='weight')

            signal_mass = [300, 420, 440, 460, 500, 600, 700, 800, 900, 1000, 1400, 1600, 2000]
            ax[4, 1].plot(signal_mass, significances)
            ax[4, 1].scatter(signal_mass, significances, c='red')
            ax[4, 2].plot(signal_mass, significances_norms)
            ax[4, 2].scatter(signal_mass, significances_norms, c='red')
            ax[4, 2].set_title('Signal Normalised')
            for axs in (ax[4, 1], ax[4, 2]):
                axs.set(xlabel='signal mass', ylabel='significance')
            fig.tight_layout()
            plt.savefig(plot_filename)
            # plt.show()
        plt.rcParams.update({'font.size': 12})
        return significances_norms

    def test(self, model_path, hidden_size, hidden_layers, plot_filename):
        input_size = 20
        activation_function = 'relu'
        output_size = 1
        model = NN(input_size, hidden_size, hidden_layers, activation_function, output_size)
        model.load_state_dict(torch.load(model_path))

        model.eval()
        with torch.no_grad():
            for t in range(5):  # Split up the test sets into 5 different sets
                significances = []
                for key, value in self.testing.items():
                    batch = value[0]
                    labels = value[1]
                    weights = value[2]
                    length = len(batch)

                    prob_weights_labels = np.zeros(int(len(value[0]) / 5), 3)

                    split_batch = batch[t * length / 5:(t + 1) * length / 5]
                    split_labels = labels[t * length / 5:(t + 1) * length / 5]
                    split_weights = weights[t * length / 5:(t + 1) * length / 5]

                    out = model(split_batch)
                    prob_weights_labels[:, 0] = out[:, 0]
                    prob_weights_labels[:, 1] = split_weights
                    prob_weights_labels[:, 2] = split_labels

                    s, b, sigma = significance(prob_weights_labels, bins=20)
                    significances.append(sigma)

