import numpy as np


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
        if b_i == 0.0:
            b_i = expected_background[:, 1].sum() / len(expected_background)  # average bkg weight
        sigma += 2 * ((s_i + b_i) * np.log(1 + (s_i / b_i)) - s_i)
    print(s, '\n', b)

    del expected_signal
    del expected_background

    return s, b, np.sqrt(sigma)