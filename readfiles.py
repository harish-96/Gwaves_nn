import glob

noise_files = glob.glob("C1/*")
signal_files = glob.glob("IFAR8/*")
params = ["netcc", "rho", "netED", "norm"]
columns = "filename,netED,norm,rho0,rho1,netcc0,netcc1,label"

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
            f_out.write(file + line[:-1] + ",1\n")
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
            f_out.write(file + line[:-1] + ",0\n")

input_file = "params.csv"

cols = pd.read_csv(input_file, nrows=1).columns
n_params = len(cols) - 2

df = pd.read_csv(input_file, usecols=cols[1:], index_col=False)
df = shuffle(df).reset_index(drop=True)

df_train = df.loc[:int(len(df)/2)-1].reset_index(drop=True)
df_test = df.loc[int(len(df)/2):].reset_index(drop=True)

x_test = df_test[cols[1:-1]].as_matrix()
y_test = df_test[cols[-1]].as_matrix()
x_train = df_train[cols[1:-1]].as_matrix()
y_train = df_train[cols[-1]].as_matrix()
