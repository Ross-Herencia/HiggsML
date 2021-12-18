import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import torch
import pickle


# transformation functions
def delphi(phi1, phi2):
    result = np.absolute(phi1 - phi2)

    mask = result > 3.14159265
    result[mask] = 2. * 3.14159265 - result[mask]

    mask = np.logical_or(phi1 < -9, phi2 < -9)
    result[mask] = -10.
    return result


def deleta(eta1, eta2):
    result = np.absolute(eta1 - eta2)

    mask = np.logical_or(eta1 < -9, eta2 < -9)
    result[mask] = -10.
    return result


def delR(phi1, phi2, eta1, eta2):
    dp = delphi(phi1, phi2)
    de = deleta(eta1, eta2)
    result = (dp ** 2 + de ** 2) ** 0.5

    mask = np.logical_or(dp < -9, de < -9)
    result[mask] = -10.
    return result


def pre_selection(df):
    dfout = df.loc[(df["region"] == 1) | (df["region"] == 2)]
    dfout = dfout.loc[dfout["regime"] == 1]
    dfout = dfout.loc[(dfout["nTags"] == 2) | (dfout['nTags'] == 3)]
    dfout.drop(columns=["nTags", "MCChannelNumber", "region", "regime", "dEtaBB", "dPhiBB"], inplace=True)
    return dfout


def post_process1(df):
    df['MV2c10B3'].replace([-99], -10, inplace=True)
    df['etaJ3'].replace([-99], -10, inplace=True)
    df['phiJ3'].replace([-99], -10, inplace=True)
    df['pTJ3'].replace([-99], -10, inplace=True)  # new: remove more -99's


def post_process2(df):
    df['dRB1B2'] = delR(df['phiB1'], df['phiB2'], df['etaB1'], df['etaB2'])
    df['dRB1J3'] = delR(df['phiB1'], df['phiJ3'], df['etaB1'], df['etaJ3'])
    df['dRB2J3'] = delR(df['phiB2'], df['phiJ3'], df['etaB2'], df['etaJ3'])
    df.drop(columns=["phiB1", "phiB2", "phiJ3", "etaB1", "etaB2", "etaJ3"], inplace=True)
    return df


# new: adding a column with the true mass of A for all instances
def add_param(df, mass):
    df['mAtrue'] = mass
    return df


def separate_masses(data_dict, signal, bkg, masses, val=True):
    for mass in masses:
        temp_signal = signal[signal['mAtrue'] == mass]
        temp_y = len(bkg) * [0] + len(temp_signal) * [1]
        temp_x = pd.concat([bkg, temp_signal], ignore_index=True)
        temp_weight = temp_x['weight'].to_numpy()
        temp_x.drop(columns=['weight'], inplace=True)
        temp_x = scaler.transform(temp_x)
        temp_x, temp_y, temp_weight = shuffle(temp_x, temp_y, temp_weight, random_state=0)
        if val == True:
            data_dict[f'val_{mass}'] = [temp_x, temp_y, temp_weight]
        else:
            data_dict[f'test_{mass}'] = [temp_x, temp_y, temp_weight]

    return data_dict


# load samples
signal_mass = [300, 420, 440, 460, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 2000]
signal = pd.DataFrame()
# filepath = 'Tong_Code_Sample\\'

for each in signal_mass:
    df_temp = pd.read_csv('C:/Users/Ross/University/Higgs Project/Raw Data/' + str(each) + ".csv", index_col=0)
    df_temp = pre_selection(df_temp)
    df_temp = add_param(df_temp, each)  # new: add signal mass

    signal = pd.concat([df_temp, signal], ignore_index=True)
df_temp = 0

background = pd.read_csv("C:/Users/Ross/University/Higgs Project/Raw Data/background.csv", index_col=0)
background = pre_selection(background)

background = add_param(background, signal_mass[np.random.randint(0, len(signal_mass))])  # new: random signal mass

# post process samples
post_process1(signal)
post_process1(background)
signal = post_process2(signal)
background = post_process2(background)

# save files
# signal.to_csv('signal.csv')
# background.to_csv('background.csv')

# remove blind mass
blind_signal = signal[signal['mAtrue'] == 1200]
signal.drop(blind_signal.index, inplace=True)

# split data
train_signal, val_test_signal = train_test_split(signal, test_size=0.4, random_state=2)
train_bkg, val_test_bkg = train_test_split(background, test_size=0.4, random_state=2)
val_signal, test_signal = train_test_split(val_test_signal, test_size=0.5, random_state=2)
val_bkg, test_bkg = train_test_split(val_test_bkg, test_size=0.5, random_state=2)

# prepare training set
train_y = len(train_bkg) * [0] + len(train_signal) * [1]
train_signal["weight"] *= np.sum(train_bkg["weight"]) / np.sum(train_signal["weight"])
train_x = pd.concat([train_bkg, train_signal], ignore_index=True)
train_weight = train_x["weight"].to_numpy()
train_x.drop(columns=["weight"], inplace=True)
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
train_x, train_y, train_weight = shuffle(train_x, train_y, train_weight, random_state=0)

# prepare mixed validation set
val_y = len(val_bkg) * [0] + len(val_signal) * [1]
val_signal["weight"] *= np.sum(val_bkg["weight"]) / np.sum(val_signal["weight"])
val_x = pd.concat([val_bkg, val_signal], ignore_index=True)
val_weight = val_x["weight"].to_numpy()
val_x.drop(columns=["weight"], inplace=True)
val_x = scaler.transform(val_x)
val_x, val_y, val_weight = shuffle(val_x, val_y, val_weight, random_state=0)

# prepare mixed test set
test_y = len(test_bkg) * [0] + len(test_signal) * [1]
test_signal['weight'] *= np.sum(test_bkg["weight"]) / np.sum(test_signal["weight"])
test_x = pd.concat([test_bkg, test_signal], ignore_index=True)
test_weight = test_x["weight"].to_numpy()
test_x.drop(columns=["weight"], inplace=True)
test_x = scaler.transform(test_x)
test_x, test_y, test_weight = shuffle(test_x, test_y, test_weight, random_state=0)

data = {}

data['train'] = [train_x, train_y, train_weight]
data['val'] = [val_x, val_y, val_weight]
data['test'] = [test_x, test_y, test_weight]

signal_mass = [300, 420, 440, 460, 500, 600, 700, 800, 900, 1000, 1400, 1600, 2000]  # not including 1200

# prepare separated validation sets
data = separate_masses(data, val_signal, val_bkg, signal_mass)

# prepare separated test sets
data = separate_masses(data, test_signal, test_bkg, signal_mass, val=False)

# prepare blind signal
blind_y = len(test_bkg) * [0] + len(blind_signal) * [1]
blind_signal['weight'] *= np.sum(test_bkg["weight"]) / np.sum(blind_signal["weight"])
blind_x = pd.concat([test_bkg, blind_signal], ignore_index=True)
blind_weight = blind_x["weight"].to_numpy()
blind_x.drop(columns=["weight"], inplace=True)
blind_x = scaler.transform(blind_x)
blind_x, blind_y, blind_weight = shuffle(blind_x, blind_y, blind_weight, random_state=0)

data['blind'] = [blind_x, blind_y, blind_weight]

# convert to tensors
for key, entry in data.items():
    i = 0
    for value in entry:
        data[key][i] = torch.Tensor(value)
        i += 1

with open('data_dict.pkl', 'wb') as file:
    pickle.dump(data, file)
