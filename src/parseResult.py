'''
Created on Jan 13, 2011

@author: kfrancoi
'''
import pickle
import os


def walkDir(fnameChunk, dirname, names):
    
    perfDict = dict()
    for name in names:
        subname = os.path.join(dirname, name)
        if os.path.isdir(subname):
            pass
        else:
            if fnameChunk in subname: 
                print subname
                file = open(subname, 'r')
                perfDict.update(pickle.load(file))
    return perfDict


if __name__ == '__main__':
    pass