import sys
from glob import glob
from pathlib import Path
from shutil import copyfile, rmtree
import numpy as np

# Example command line:
#   python -m deepteam.sampler /data/deepteam/quads-1x1-50k-meta50k /data/deepteam/quads-1x1-50k-samples-5k 5000 20 15 15 15 15 10 10
# python -m deepteam.sampler /data/deepteam/fulldata-meta /data/deepteam/fulldata-samps 30000 20 15 15 15 15 10 10

inDir= sys.argv[1]
outDir = sys.argv[2]
print(f'Reading from {inDir} and writing to {outDir}') #  and filtering out for y<{maxy}')
rmtree(outDir, ignore_errors=True)
Path(outDir).mkdir(parents=True,exist_ok=True)

def parseQuads(fn):
    with open(fn,'r') as f:
        # fns = [f'{inDir}/{x.strip()}' for x in f.readlines()]
        fns = [f'{x.strip()}' for x in f.readlines()]
        return fns

totalFiles = int(sys.argv[3])

NumQuads = len(sys.argv) - 4
quads = [int(sys.argv[i]) for i in range(4,4+NumQuads)]
pcts = [ quads[i]/sum(quads) for i in range(len(quads))]
numFiles = [int(pct * totalFiles) for pct in pcts]

filesPerQuad = [parseQuads(f'{inDir}/quads{i}.txt') for i in range(NumQuads)]
nFilesPerQuad = [len(fq) for fq in filesPerQuad]

for i in range(len(nFilesPerQuad)):
    if nFilesPerQuad[i] < numFiles[i]:
        print(f'WARN: we do not have enough images for quadrant[{i}] (requested={numFiles[i]}) so resetting to {nFilesPerQuad[i]}')
        numFiles[i] = nFilesPerQuad[i]

fileNames = [np.random.choice(filesPerQuad[i], size=int(numFiles[i]),replace=False)
                    for i in range(len(quads))]

metafn = f'{outDir}/summary.txt'
with open(metafn,'w') as metaf:
    cnt = 0
    for i in range(len(fileNames)):
        fn = f'{outDir}/quad_filenames{i}.txt'
        msg = f'For Quadrant{i}: writing {len(fileNames[i])} filenames to {fn}..'
        print(msg)
        metaf.write(f'{msg}\n')
        with open(fn,'w') as f:
            f.write('\n'.join(fileNames[i]))
        cnt+= len(fileNames[i])
    msg=f'Total filenames written: {cnt}'
    print(msg)
    metaf.write(msg + '\n')
print(f'Wrote summary to {metafn}')

# copy the files
print(f'Copying csvs from {inDir} to {outDir} ..')
fcnt = 0
for files in fileNames:
    for fn in files:
        fp = fn[fn.rfind('/')+1:]
        copyfile(fn, f"{outDir}/{fp}")
        fcnt+=1
    print(f'Progress: {fcnt} files copied..')
    
print(f'Copied {fcnt} csvs from {inDir} to {outDir}')
