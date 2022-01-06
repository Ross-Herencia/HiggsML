import numpy as np
import matplotlib.pyplot as plt


def significance(prob_weights_labels):
    s = []
    b = []
    sigma = 0.0
    signal_mask_label = prob_weights_labels[:, 2] == 1.0
    background_mask_label = prob_weights_labels[:, 2] == 0.0
    expected_signal = prob_weights_labels[signal_mask_label]
    expected_background = prob_weights_labels[background_mask_label]

    signal_mask_prob = (expected_signal[:, 0] >= 0.0) & (expected_signal[:, 0] <= 0.05)
    background_mask_prob = (expected_background[:, 0] >= 0.0) & (expected_background[:, 0] <= 0.05)
    s.append(expected_signal[signal_mask_prob][:, 1].sum())
    b.append(expected_background[background_mask_prob][:, 1].sum())

    for i in np.arange(0.05, 1.0, 0.05):
        signal_mask_prob = (expected_signal[:, 0] > i) & (expected_signal[:, 0] <= i + 0.1)
        background_mask_prob = (expected_background[:, 0] > i) & (expected_background[:, 0] <= i + 0.1)
        s.append(expected_signal[signal_mask_prob][:, 1].sum())
        b.append(expected_background[background_mask_prob][:, 1].sum())

    for s_i, b_i in zip(s, b):
        if (s_i != 0.0) & (b_i == 0.0):
            b_i = expected_background[:, 1].sum() / len(expected_background)  # average bkg weight
            sigma += 2 * ((s_i + b_i) * np.log(1 + (s_i / b_i)) - s_i)
        elif (s_i == 0.0) & (b_i == 0.0):
            sigma += 0.0
        else:
            sigma += 2 * ((s_i + b_i) * np.log(1 + (s_i / b_i)) - s_i)

    # print(s, '\n', b)

    del expected_signal
    del expected_background

    return s, b, np.sqrt(sigma)


def plot_loss(loss_list, legend_labels, log_scale=False, plot_name='loss'):
    n = 1
    for entry in loss_list:
        history = []
        for epoch, value in entry:
            history.append(value)

        plt.plot(history, label=legend_labels[n-1])
        n += 1
    if log_scale:
        plt.yscale('log')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    # plt.show()
    plt.savefig(f'{plot_name}.png')


def plot_significance_dist(s, b, sigma, line=False, bar=True):
    x = np.arange(0, 1, 1/len(s))
    if bar:
        plt.bar(x, s, width=1/len(s), align='edge', label='signal', color='red', alpha=0.7)
        plt.bar(x, b, width=1/len(s), align='edge', label='background', color='blue', alpha=0.7)
    if line:
        plt.plot(x, s, label='signal', color='red')
        plt.plot(x, b, label='background', color='blue')
    plt.xlabel('probability')
    plt.ylabel('significance')
    plt.legend()
    plt.title(f'sigma = {sigma}')
    plt.savefig(f'significance_dist_{sigma}.png')
    # plt.show()


def plot_significance_curve(path, significances):
    signal_mass = [300, 420, 440, 460, 500, 600, 700, 800, 900, 1000, 1400, 1600, 2000]
    plt.plot(signal_mass, significances)
    plt.xlabel('signal mass')
    plt.ylabel('significance')
    plt.savefig(path)
