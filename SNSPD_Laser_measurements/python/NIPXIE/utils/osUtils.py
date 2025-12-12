#!/usr/bin/env python3

import os
import warnings
import os
import errno


def create_755(path):
    createDir(path)
    os.chmod(path,0o755)

def rmDir(path):
    os.rmdir(path)

def createDir(Dir):
    try:
        os.makedirs(Dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
            # print(f'{Dir} exists.')
        else:
            raise
