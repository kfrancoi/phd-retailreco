'''
Created on Feb 14, 2011

@author: kfrancoi
'''
#import getopt
#import sys
#import os
#import time
#import datetime
#import matplotlib.pyplot as plt
#import processing3 as processing
#import pickle
#from numpy import *
#from scipy.io import mmwrite, mmread
from Experiences import *

#from model import *
dataFolder = '../data/'
resultFolder = '../result/Kristof/' 


data = loadPyPred()


modelCosine = processing.CosineRecoModel(data.getUserItemMatrix())

#Cosine
evalCosine = processing.Evaluation(modelCosine, data.getBasketItemList(), 10)
recoItems = evalCosine.recommendAll('novel')
evalCosine.writeRecomendation(recoItems, data.basketDict, data.itemDict, resultFolder+'Cosine_novel.txt')

recoItems = evalCosine.recommendAll('all')
evalCosine.writeRecomendation(recoItems, data.basketDict, data.itemDict, resultFolder+'Cosine_all.txt')

#Randow walk with restart
modelRW = processing.BasketRandomWalk(data.getUserItemMatrix(), 0.1)

evalRW = processing.Evaluation(modelRW, data.getBasketItemList(), 10)
recoItems = evalRW.recommendAll('novel')
evalRW.writeRecomendation(recoItems, data.basketDict, data.itemDict, resultFolder+'RW_novel_alpha_01.txt')

#SOP
modelSOP = processing.SOPRecoModel(data.getUserItemMatrix(), 0.95, 2)
modelSOP.train()

evalSOP = processing.Evaluation(modelSOP, data.getBasketItemList(), 10)
recoItems = evalSOP.recommendAll('novel')
evalSOP.writeRecomendation(recoItems, data.basketDict, data.itemDict, resultFolder+'SOP_novel_alpha_095_t2.txt')

