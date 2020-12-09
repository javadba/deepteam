import os
from glob import glob
import numpy as np

# dir = '/Users/steve/git/deepteam/final/2x2x250'
import sys
dir = sys.argv[1] #
dir = '/Users/steve/git/deepteam/final/3xrandomagentx1250z'

x,y,z=[],[],[]
for fn in glob(f'{dir}/*.csv'):
    with open(fn,'r') as f:
        dats = f.readlines()[0]
        xyz = list(map(lambda x: float(x),dats.split(',')))
        x.append(xyz[0])
        y.append(xyz[1])
        z.append(xyz[2])
print(np.mean(z))

