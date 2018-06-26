import glob
import numpy as np

files = glob.glob("IFAR8/*")
prefix = 'IFAR8_separated/'
for file in files:
    with open(file, 'r') as f:
        d = f.readlines()
        splits = [i for i in range(len(d)) if "trigger" in d[i]]
        for j in range(len(splits)-1):
            g = open(prefix + file[6:-4] + '_part' + str(j) + '.txt', 'w')
            g.write(''.join(d[splits[j]:splits[j+1]]))
            g.close()
        j = len(splits)-1
        g = open(prefix + file[6:-4] + '_part' + str(j) + '.txt', 'w')
        g.write(''.join(d[splits[j]:]))
        g.close()

