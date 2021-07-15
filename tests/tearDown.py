import glob
import shutil
import sys,os

envname = sys.argv[1]
log_dir = os.path.normpath(os.path.join(sys.argv[2],envname))

print('Saving aares logs to: {}'.format(log_dir))

os.makedirs(log_dir,exist_ok=True)
print(os.getcwd())
for fi in glob.glob('*.log'):
    shutil.copy(fi,log_dir)