from glob import glob
import sys
from shutil import copyfile
from pathlib import Path

dir1, dir2,outDir = sys.argv[1], sys.argv[2], sys.argv[3]

Path.mkdir(Path(outDir),parents=True,exist_ok=True)

dir1f = set([f[f.find('/')+1:] for f in glob(f'{dir1}/*')])
dir2f = set([f[f.find('/')+1:] for f in glob(f'{dir2}/*')])


diff = dir1f.difference(dir2f)
for f in diff:
    copyfile(f'{dir1}/{f}',f'{outDir}/{f}')

