import numpy as np


def significance(prob_weights_labels, bins=10, normalise=False, remove_negative_weights=False, make_positive=False):
    s = []
    b = []
    sigma = 0.0
    signal_mask_label = prob_weights_labels[:, 2] == 1.0
    background_mask_label = prob_weights_labels[:, 2] == 0.0
    expected_signal = prob_weights_labels[signal_mask_label]
    expected_background = prob_weights_labels[background_mask_label]

    if remove_negative_weights:
        bkg_pos_mask = expected_background[:, 1] >= 0.0
        expected_background = expected_background[bkg_pos_mask]

    if make_positive:
        expected_background[:, 1] = np.abs(expected_background[:, 1])

    if normalise:
        expected_signal[:, 1] = expected_signal[:, 1] / expected_signal[:, 1].sum()  # Normalisation
        expected_background[:, 1] = expected_background[:, 1] / expected_background[:, 1].sum()  #

    bin_size = 1 / bins

    signal_mask_prob = (expected_signal[:, 0] >= 0.0) & (expected_signal[:, 0] <= bin_size)
    background_mask_prob = (expected_background[:, 0] >= 0.0) & (expected_background[:, 0] <= bin_size)
    s.append(expected_signal[signal_mask_prob][:, 1].sum())
    b.append(expected_background[background_mask_prob][:, 1].sum())

    for i in np.arange(bin_size, 1, bin_size):
        signal_mask_prob = (expected_signal[:, 0] > i) & (expected_signal[:, 0] <= i + bin_size)
        background_mask_prob = (expected_background[:, 0] > i) & (expected_background[:, 0] <= i + bin_size)
        s.append(expected_signal[signal_mask_prob][:, 1].sum())
        b.append(expected_background[background_mask_prob][:, 1].sum())

    for s_i, b_i in zip(s, b):
        if (s_i != 0.0) & (b_i == 0.0):
            b_i = expected_background[:, 1].sum() / len(expected_background)  # average bkg weight
            sigma += 2 * ((s_i + b_i) * np.log(1 + (s_i / b_i)) - s_i)
        elif (s_i == 0.0) & (b_i == 0.0):
            sigma += 0.0
        elif b_i < 0:
            b_i = np.abs(b_i)
            sigma += 2 * ((s_i + b_i) * np.log(1 + (s_i / b_i)) - s_i)
        else:
            sigma += 2 * ((s_i + b_i) * np.log(1 + (s_i / b_i)) - s_i)

    del expected_signal
    del expected_background

    return s, b, np.sqrt(sigma)

