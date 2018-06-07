import glob
import pandas as pd
from sklearn.utils import shuffle

noise_files = glob.glob("C1/*")
signal_files = glob.glob("IFAR8/*")
params = ["netcc", "rho", "netED", "norm"]
columns = ["netED","norm","rho0","rho1","netcc0","netcc1","label"]

n_params = len(columns.split(","))

df = pd.DataFrame(columns=columns)

for it,file in enumerate(signal_files):
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
                line += tmp
        df.loc[it] = line + [1]

for it,file in enumerate(noise_files):
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
                line += tmp
        df.loc[it+len(signal_files)] = line + [0]

n_params = len(columns) - 1

df = shuffle(df).reset_index(drop=True)

df_train = df.loc[:int(len(df)/2)-1].reset_index(drop=True)
df_test = df.loc[int(len(df)/2):].reset_index(drop=True)

df_train.to_csv("train_data.csv")
df_test.to_csv("test.csv")
