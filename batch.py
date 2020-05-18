import os

filelst = os.listdir('./')
for filepath in filelst:
    if os.path.isdir(filepath):
        b = os.listdir(filepath)
        for filename in b:
            if filename.endswith('model'):
                cmd = './convert.sh %s/%s %s' % (filepath, filename, filepath)
                print(cmd)
                os.system(cmd)
