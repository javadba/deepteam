import sys
from glob import glob
from pathlib import Path
from shutil import copyfile, rmtree
import numpy as np

# Example command line:
#   python -m deepteam.sampler /data/deepteam/quads-1x1-50k-meta50k /data/deepteam/quads-1x1-50k-samples-5k 5000 20 15 15 15 15 10 10
# python -m deepteam.sampler /data/deepteam/fulldata-meta /data/deepteam/fulldata-samps 30000 20 15 15 15 15 10 10

inDir= sys.argv[1]
nImgs = int(sys.argv[2])

print(f'Reading {nImgs} from {inDir}')

model = 
for fn in glob(f'{inDir}/*.csv'):
    with open(fn,'r') as f:
    
