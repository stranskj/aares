import os
import sys

try:
    os.symlink(sys.argv[1],'data',target_is_directory=True)
except OSError as e:
    print(e)
    print('WARNING: Probably insufficient permissions to create symbolic link in Windows. Running as administrator might help. You can try to follow this guide: https://stackoverflow.com/questions/6260149/os-symlink-support-in-windows')
    sys.exit(1)