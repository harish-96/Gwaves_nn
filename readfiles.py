import glob
import pandas as pd
from sklearn.utils import shuffle

noise_files = glob.glob("C1/*")
signal_files = glob.glob("IFAR8/*")
params = ["netcc", "rho", "netED", "norm"]
columns = "netED,norm,rho0,rho1,netcc0,netcc1,label"

n_params = len(columns.split(","))

with open("params.csv", "w") as f_out:
    f_out.write(columns + "\n")
    for file in signal_files:
        with open(file, "r") as f:
            d = f.readlines()
            line = ""
            for i in range(2, 72):
                name = d[i].split()[0][:-1]
                if name in params:
                    tmp = d[i].split()[1:]
                    if name == "rho" and i < 10:
                        tmp = []
                    if name == "netcc":
                        tmp = d[i].split()[1:3]
                    line += ",".join(tmp) + ","
            f_out.write(line[1:-1] + ",1\n")
    for file in noise_files:
        with open(file, "r") as f:
            d = f.readlines()
            line = ""
            for i in range(2, 72):
                name = d[i].split()[0][:-1]
                if name in params:
                    tmp = d[i].split()[1:]
                    if name == "rho" and i < 10:
                        tmp = []
                    if name == "netcc":
                        tmp = d[i].split()[1:3]
                    line += ",".join(tmp) + ","
            f_out.write(line[1:-1] + ",0\n")

input_file = "params.csv"

cols = pd.read_csv(input_file, nrows=1).columns
df = pd.read_csv(input_file, index_col=False)
df = shuffle(df).reset_index(drop=True)

df_train = df.iloc[:int(len(df)/2)].reset_index(drop=True)
df_test = df.iloc[int(len(df)/2):].reset_index(drop=True)

df_test.to_csv("test_data.csv", index=False)
df_train.to_csv("train_data.csv", index=False)
