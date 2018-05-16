import glob
import pandas as pd
from sklearn.utils import shuffle
import numpy as np

noise_files = glob.glob("C1/*")
signal_files = glob.glob("IFAR8/*")
params = ["rho","norm","netcc","snr"]
cols = ["filename","norm","rho0", "rho1","snr0","snr1","netcc0","netcc1","label"]

data = []

for file in signal_files:
    with open(file, "r") as f:
        d = f.readlines()
        line = []
        for i in range(2, 72):
            name = d[i].split()[0][:-1]
            if name in params:
                tmp = d[i].split()[1:]
                if name == "rho" and i < 10:
                    tmp = []
                if name == "netcc":
                    tmp = d[i].split()[1:3]
                line = np.append(line, tmp)
        data.append([file] + list(line) + [1])
for file in noise_files:
    with open(file, "r") as f:
        d = f.readlines()
        line = []
        for i in range(2, 72):
            name = d[i].split()[0][:-1]
            if name in params:
                tmp = d[i].split()[1:]
                if name == "rho" and i < 10:
                    tmp = []
                if name == "netcc":
                    tmp = d[i].split()[1:3]
                line = np.append(line, tmp)
        data.append([file] + list(line) + [0])

df = pd.DataFrame(data, columns=cols)
df = shuffle(df).reset_index(drop=True)

df_train = df.iloc[:int(len(df)/2)].reset_index(drop=True)
df_test = df.iloc[int(len(df)/2):].reset_index(drop=True)

df_test.to_csv("5params_test_data.csv", index=False)
df_train.to_csv("5params_train_data.csv", index=False)
