from glob import glob
import os

os.chdir('scrubbed31kNoYFilter')
fl1 = glob('*.png')
fl1 = set([f[0:f.find('.')] for f in fl1])

fl2 = glob('*.csv')
fl2 = set([f[0:f.find('.')] for f in fl2])

diff = fl1.difference(fl2)
print(f"len(fl1)={len(fl1)} len(fl2)={len(fl2)} diff is {diff}")

diff = fl2.difference(fl1)
print(f"diff2 is {diff}")

