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
import scipy.io as io
from numpy import *


#PARAM
nbReco = 3
nbBasket = 500

trainUIcvFNAME = 'UI.mtx'
valBasketcvFNAME = 'testBaskets.txt'


#Create an array with result from every methods
print('Prepare dataset')
preprocessor = processing.TransactionPreprocessing()
if os.path.exists(trainUIcvFNAME) or os.path.exists(valBasketcvFNAME):
    
    testBaskets = preprocessor.loadBasketData(valBasketcvFNAME)
    UI = preprocessor.loadTrainData(trainUIcvFNAME)

else :
    #READ TRAIN DATASET PARAM
    dataPaths = ['../data/TaFengDataSet/D11', '../data/TaFengDataSet/D12', '../data/TaFengDataSet/D01']
    delimiter = ';'
    skiprows = 1
    usecols = (0,1,4)
    event_fields = ['datetime', 'customerID', 'item']
    event_format = ['S20', 'int', 'int']
    dt=dtype({'names': [x for x in event_fields], 'formats': [b for b in event_format]})
    
    
    transData = preprocessor.readDataFile(dataPaths, delimiter, skiprows, usecols, dt)
    
    UI = preprocessor.aggregateUserProcessing(transData)
    preprocessor.saveTrainData(trainUIcvFNAME, UI)
    
    #READ TEST DATASET PARAM
    dataPaths = ['../data/TaFengDataSet/D02']
    testData = preprocessor.readDataFile(dataPaths, delimiter, skiprows, usecols, dt)
    
    testBaskets = preprocessor.aggregateTransactionProcessing(testData)
    preprocessor.saveBasketData(valBasketcvFNAME, testBaskets)


#OPTIMIZE SOP
mf = core.ModelFactoryReco()
jobs = list()
for alpha, teta in [(1,1)]:
    modelSOP = processing.SOPRecoModel(UI, alpha, teta)
    if nbBasket == -1:
        evalSOP = processing.Evaluation(modelSOP, testBaskets, nbReco)
    else :
        evalSOP = processing.Evaluation(modelSOP, testBaskets[:nbBasket], nbReco)
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

