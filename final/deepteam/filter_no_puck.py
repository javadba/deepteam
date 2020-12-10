import sys
from glob import glob
from pathlib import Path
from shutil import copyfile, rmtree
import numpy as np

inDir= sys.argv[1]
outDir = sys.argv[2]
maxFiles = int(sys.argv[3]) if len(sys.argv)>=4 else sys.maxsize
copyImgs = sys.argv[4].lower()=='true' if len(sys.argv)>=5 else True

print(f'Reading from {inDir} and writing to {outDir}') #  and filtering out for y<{maxy}')
if False:
    rmtree(outDir, ignore_errors=True)
Path(outDir).mkdir(parents=True,exist_ok=True)
FarY = -0.3
offscreen =  lambda x,y: int(abs(x)>=0.999) #  or y>0.5)
quadrules = [
    ['offscreen', offscreen],
    ['nearcenterx-fary', lambda x,y: int(not offscreen(x,y) and abs(x) <= 0.2 and y <= FarY )],
    ['offcenterx-fary',  lambda x,y: int(not offscreen(x,y) and abs(x)>0.2 and abs(x) <= 0.4 and y <= FarY)],
    ['offcenterx-fary',  lambda x,y: int(not offscreen(x,y) and abs(x)>0.4 and y <= FarY)],
    ['nearcenterx-neary',lambda x,y: int(not offscreen(x,y) and abs(x) <= 0.3 and y > FarY)],
    ['offcenterx-neary', lambda x,y: int(not offscreen(x,y) and abs(x) > 0.3 and abs(x) <= 0.5 and y > FarY)],
    ['farx-neary',       lambda x,y: int(not offscreen(x,y) and abs(x) >0.5 and y > FarY)]
]

quadfiles = [[] for i in range(len(quadrules))] # [''] * 6
for i,fn in ((i,fn) for (i,fn) in enumerate(glob(f'{inDir}/*.csv')) if i<maxFiles):
    with open(fn,'r') as f:
        fn = fn[fn.rfind('/')+1:]
        lines = [ll.strip() for ll in f.readlines()]
        x,y = [float(z)
               for ll in lines
                   for z in ll.split(',')]
        if offscreen(x,y):
            x = -1
            y = -1
        
        ofn = f'{outDir}/{fn}'
        with open(ofn, 'w') as of:
            of.write(f'{str(x)},{str(y)},')
            quads = [qr[1](x,y) for qr in quadrules]
            for i,q in ((i,q) for (i,q) in enumerate(quads) if q):
                quadfiles[i].append(f'{outDir}/{fn}')
            quadstr = [str(q) for q in quads]
            of.write(','.join(quadstr))
        png =f"{fn[0:fn.find('.')]}.png"
        if copyImgs:
            copyfile(f"{inDir}/{png}",f"{outDir}/{png}")

metafn = f'{outDir}/summary.txt'
with open(metafn,'w') as metaf:
    cnt = 0
    for i in range(len(quadrules)):
        fn = f'{outDir}/quads{i}.txt'
        msg = f'For {quadrules[i][0]}:\t\t\twriting {len(quadfiles[i])} filenames to {fn}..'
        print(msg)
        metaf.write(f'{msg}\n')
        with open(fn,'w') as f:
            f.write('\n'.join(quadfiles[i]))
        cnt+= len(quadfiles[i])
    msg=f'Total filenames written: {cnt}'
    print(msg)
    metaf.write(msg + '\n')
print(f'Wrote summary to {metafn}')