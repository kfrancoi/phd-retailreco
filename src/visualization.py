'''
Created on Jan 6, 2011

@author: kfrancoi
'''
from numpy import *
import matplotlib.pyplot as plt

def modelPerfLine2D(data, title, paramX, perfY):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data[:,0], data[:,1])
    ax.set_title(title)
    ax.set_xlabel(paramX)
    ax.set_ylabel(perfY)
    plt.savefig(title)
    
    
def itemWeight(weightList, sort=0):
        print 'Item weight'
        if sort != 0:
            ind = flipud(argsort(weightList))
            plt.bar(range(len(weightList)),weightList[ind])
        else :
            plt.bar(range(len(weightList)),weightList)
        plt.show()
        

if __name__=='__main__':
    noveltySOP = array([[0, 6.645e-01],
                  [0.5, 0.641],
                  [1, 0.591]])
    
    popularitySOP = array([[0, 4.717e-04],
                           [0.5, 0.0007],
                           [1, 0.0015]])
    
    bhrPopSOP = array([[0, 8.4e-02],
                       [0.5, 0.1],
                       [1, 0.126]])
    
    whrSOP = array([[0, 6e-02],
                    [0.5, 0.03],
                    [1, 0.075]])
    
    modelPerfLine2D(noveltySOP, 'SOP Novelty rate', 'alpha', 'novelty')
    modelPerfLine2D(popularitySOP, 'SOP Popularity rate', 'alpha', 'pop')
    modelPerfLine2D(bhrPopSOP, 'SOP Binary Hit Rate', 'alpha', 'bhrPop')
    modelPerfLine2D(whrSOP, 'SOP Weighted Hit Rate', 'alpha', 'whr')