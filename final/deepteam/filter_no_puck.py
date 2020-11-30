import os,sys
from glob import glob
from shutil import copyfile, rmtree
from pathlib import Path

maxx, maxy = float(sys.argv[1]),float(sys.argv[2])
outDir = sys.argv[3]
inDir= 'save_data_dir' if len(sys.argv) < 5 else sys.argv[4]

print(f'Writing to {outDir} and filtering out for y<{maxy}')
rmtree(outDir, ignore_errors=True)
Path(f'/git/deepteam/final/{outDir}').mkdir(parents=True,exist_ok=True)

for fn in glob(f'{inDir}/*.csv'):
    with open(fn,'r') as f:
        fn = fn[fn.find('/')+1:]
        lines = [ll.strip() for ll in f.readlines()]
        x,y = [float(z)
               for ll in lines
                   for z in ll.split(',')]
        # if abs(y) >= 0.9 and abs(y) < 1.0:
        #     print(y)
        if abs(x)<maxx and y<maxy:
            ofn = f'{outDir}/{fn}'
            with open(ofn, 'w') as of:
                of.write(f'{str(x)},{str(y)}')
            png =f"{fn[0:fn.find('.')]}.png"
            copyfile(f"{inDir}/{png}",f"{outDir}/{png}")
        else:
            print(f'Filtering out ({x},{y})..')
    