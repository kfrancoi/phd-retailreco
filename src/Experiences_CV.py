'''
Created on Jan 4, 2011

@author: kfrancoi
'''
import os
import sys
import datetime
import core
import processing3 as processing
import Experiences
import pickle
import scipy.io as io
from numpy import *


#PARAM
nbReco = 3
nbBasket = 500


def cvPythProcessing():
    raise NotImplementedError

def cvTaFengProcessing():
    
    trainUIcvFNAME = 'UIcv.mtx'
    valBasketcvFNAME = 'valBasketscv.txt'
    
    
    #Create an array with result from every methods
    print('Prepare dataset')
    preprocessor = processing.TransactionPreprocessing()
    if os.path.exists(trainUIcvFNAME) or os.path.exists(valBasketcvFNAME):
        
        valBaskets = preprocessor.loadBasketData(valBasketcvFNAME)
        UIcv = preprocessor.loadTrainData(trainUIcvFNAME)
    
    else :
        #READ DATASET PARAM
        dataPaths = ['../data/TaFengDataSet/D11', '../data/TaFengDataSet/D12', '../data/TaFengDataSet/D01', '../data/TaFengDataSet/D02']
        delimiter = ';'
        skiprows = 1
        usecols = (0,1,4)
        event_fields = ['datetime', 'customerID', 'item']
        event_format = ['S20', 'int', 'int']
        dt=dtype({'names': [x for x in event_fields], 'formats': [b for b in event_format]})
        
        
        transData = preprocessor.readDataFile(dataPaths, delimiter, skiprows, usecols, dt)
        
        #Use only the first three month for training
        trainTransData = list()
        testTransData = list()
        for trans in transData:
            year = int(trans[0].split(' ')[0].split('-')[0])
            month = int(trans[0].split(' ')[0].split('-')[1])
            day= int(trans[0].split(' ')[0].split('-')[2])
            if datetime.datetime(year, month, day) < datetime.datetime(2001, 2, 1):
                trainTransData.append(trans)
            else :
                testTransData.append(trans)
        
        
        #Use only train set to get best parameters
        trainTransDataCV = trainTransData[:len(trainTransData)*3/4]
        valTransDataCV = trainTransData[len(trainTransData)*3/4:]
        
        UIcv = preprocessor.aggregateUserProcessing(trainTransDataCV)
        preprocessor.saveTrainData('UIcv', UIcv)
        valBaskets = preprocessor.aggregateTransactionProcessing(valTransDataCV)
        preprocessor.saveBasketData('valBaskets.txt', valBaskets)
    
    return UIcv, valBaskets

UIcv, valBaskets = cvTaFengProcessing()

mf = core.ModelFactoryReco(2)
jobs = list()
for alpha, teta in [(0.1,0.1), (0.1,0.5), (0.1,1), (0.5, 0.1), (0.5, 0.5), (0.5, 1), (0.95, 0.1), (0.95, 0.5), (0.95, 1), (1,2)]:
    modelSOP = processing.SOPRecoModel(UIcv, alpha, teta)
    if nbBasket == -1:
        evalSOP = processing.Evaluation(modelSOP, valBaskets, nbReco)
    else :
        evalSOP = processing.Evaluation(modelSOP, valBaskets[:nbBasket], nbReco)
    jobs.append(mf.add(modelSOP, evalSOP))

mf.start()

#Check result
results = list()
for job in jobs:
    results.append(mf.get(job))

#Get best params
perfTester = processing.Evaluation()
SOPPerf = dict()
for job in results:
    for performance in perfTester.testNames: 
        SOPPerf[(performance, job.model.alpha, job.model.teta)] = job.evaluator.meanPerf[[i for i,j in enumerate(perfTester.testNames) if j==performance]][0]

print 'Writing Baskets to file'
file = open('modelPerf.txt', 'w')
pickle.dump(SOPPerf, file)
file.close()    