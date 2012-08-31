'''
Created on May 11, 2010

@author: kfrancoi
'''
import getopt
import sys
import os
import time
import datetime
import matplotlib.pyplot as plt
import processing_Multi as processing
import pickle
from numpy import *
from scipy.io import mmwrite, mmread

#from model import *
dataFolder = '../data/'
resultFolder = '../result2/' 

#result2 --> cos
#result --> bn

#def toyNetwork():
#	import Graph
#	G = Graph.Graph()
#	G.readDWalkAdj('data/roadmap2.txt')
#	return G.adjacency

def testVisu2(UI, evidence=[0,3], label=''):
	
	alpha = [0.01,0.05,0.1,0.3,0.7,0.95]
	teta = [0, 1, 2, 5, 10]
	for a in alpha:
		plt.figure(figsize=(20,20))
		count=1
		for t in teta:
			print 'Recommendation model for alpha %s and teta %s'%(a,t)
			model = processing.SOPRecoModel(UI, a, t)
			model.launch()
			try :
				b = model.getScores(evidence, -1)
			except linalg.LinAlgError:
				print 'Alpha %s and teta %s produce a Singular matrix exception'%(a,t)
				continue
			plt.subplot(len(teta),1,count)
			count+=1
			
			popInd = flipud(argsort(model.ItemPop))
			plt.bar(range(len(b[popInd])),b[popInd])
			
			bInd = argsort(b[:])
			
			plt.title('alpha %s teta %s : %s'%(a,t,bInd[-3:]))
		plt.savefig(resultFolder+'TestBasket_SOP %s alpha %s.pdf'%(label,a))

def testCosine(UI, evidence=[0,3], label=''):
	
	print 'Recommendation model'
	model = processing.CosineRecoModel(UI)
	b = model.getScores(evidence, -1)
	popInd = flipud(argsort(model.ItemPop))
	plt.bar(range(len(b[popInd])),b[popInd])
	bInd = argsort(b[:])
	plt.title('Cosine : %s'%bInd[-3:])
	plt.savefig(resultFolder+'TestBasket_Cosine %s.pdf'%(label))

def loadToy():
	
	data = processing.TransactionPreprocessing()
	
	UI = array([
			[1,1,0,1,0,0],
			[1,1,0,0,1,0],
			[1,0,1,1,0,0],
			[1,1,1,0,0,0],
			[1,0,1,0,0,0],
			[1,1,0,1,0,0],
			[1,1,0,0,1,0],
			[1,0,1,1,0,0],
			[1,1,1,0,0,1],
			[0,0,1,1,0,1],
			], dtype='float')
	
	data.UI = UI
	
	data.BasketItemList = list()
	
	data.BasketItemList.append(array([0,1,3]))
	data.BasketItemList.append(array([0,1,4]))
	data.BasketItemList.append(array([0,2]))
	data.BasketItemList.append(array([0,1,4]))
	data.BasketItemList.append(array([2,3,4]))
	
	return data
	

def load():
	
	###############################################################
	# PARSE DATA
	###############################################################
	print 'Parse Data'
	path = dataFolder+'TaFengDataSet/'
	dataPaths = [path+'D01', path+'D11', path+'D12']
	data = processing.BasketPreprocessing()
	if os.path.exists(data.saveUI):
		print 'Train data exists --> loading files...'
		data.loadTrainData()
	else:
		print 'Train data does not exist --> parsing files...'
		data.trainDataPreprocessing(dataPaths)
		data.saveTrainData()
	
	###############################################################
	# LOAD BASKET DATA
	###############################################################
	if os.path.exists(data.saveBI):
		print 'Basket data exists --> loading files...'
		data.loadBasketData()
	else :
		print 'Basket data does not exist --> parsing files...'
		data.testDataPreprocessing(path+'D02')
		data.saveBasketData()

	return data

def loadPyPred():
	
	trainUIPyPred = 'UIPyPred.mtx'
	testBasketPyPred = 'testBasketPyPred.txt'
	testBasketCustomerID = 'testBasketCustomerID.txt'
	itemDictFile = 'itemDict.txt'
	
	print 'Parse Data'
	path = dataFolder+'pyPred/'
	preprocessor = processing.TransactionPreprocessing()
	if os.path.exists(trainUIPyPred) and os.path.exists(testBasketPyPred):
		testBasket = preprocessor.loadBasketData(testBasketPyPred)
		UI = preprocessor.loadTrainData(trainUIPyPred)
		preprocessor.loadCustomerIDDict(testBasketCustomerID)
		preprocessor.loadItemIDDict(itemDictFile)
	else:
		dataPath = [path+'train_invraw3.txt']
		delimiter = ','
		skiprows = 1
		usecols = (0,1)
		event_fields = ['customer_id', 'item_id']
		event_format = ['int', 'int']
		dt=dtype({'names': [x for x in event_fields], 'formats': [b for b in event_format]})
		
		#Parse Data
		transData = preprocessor.readDataFile(dataPath, delimiter, skiprows, usecols, dt)
		
		#Aggregate Data
		UI = preprocessor.aggregateUserProcessing2C(transData)
		preprocessor.saveTrainData('UIPyPred', UI)
		
		#Record transaction Data
		dataPath = [path+'test_invraw3.txt']
		delimiter = ','
		skiprows = 1
		usecols = (0,1)
		event_fields = ['customer_id', 'item_id']
		event_format = ['int', 'int']
		dt=dtype({'names': [x for x in event_fields], 'formats': [b for b in event_format]})
		transDataTest = preprocessor.readDataFile(dataPath, delimiter, skiprows, usecols, dt)
		testBasket = preprocessor.aggregateTransactionProcessing2C(transDataTest)
		preprocessor.saveBasketData(testBasketPyPred, testBasket)
		preprocessor.saveCustomerIDDict(testBasketCustomerID)
		preprocessor.saveItemIDDict(itemDictFile)
	
	return preprocessor
	
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
	
	return preprocessor


def SOP(alpha, teta, nbBasket, nbReco):
	
	data = load()
	#data = loadPyPred()
	#data = cvTaFengProcessing()
	###############################################################
	# CREATE MODELS
	###############################################################
	print 'Create the model based on the training set'
	
	modelSOP = processing.SOPRecoModel(data.getUserItemMatrix(), alpha, teta)
	modelSOP.train()
		
	###############################################################
	# SET RECOMMENDATION
	###############################################################
	if nbBasket == -1:
		evalSOP = processing.Evaluation(modelSOP, data.getBasketItemList(), nbReco)
	else :
		evalSOP = processing.Evaluation(modelSOP, data.getBasketItemList()[:nbBasket], nbReco)
	
	###############################################################
	# LAUNCH RECOMMENDATION + SAVE RESULTS
	###############################################################	
	t = time.time()
	evalSOP.newEval()
	SOPTime = time.time()-t
	
#	session = getSession()
#	model = Model('SOP')
#	model.parameters.append(ModelParameters(getParameter('alpha'), alpha))
#	model.parameters.append(ModelParameters(getParameter('teta'), teta))
#	model.parameters.append(ModelParameters(getParameter('nbBasket'), nbBasket))
#	model.parameters.append(ModelParameters(getParameter('nbReco'), nbReco))
#	mmwrite(resultFolder+'SOPPerf_a%s_t%s_nb%s_nr%s'%(alpha,teta,nbBasket,nbReco),evalSOP.perf) 
	
	print 'SOP Execution time:', SOPTime
	print 'Performances : '
	print evalSOP.testNames
	print evalSOP.computePerf()
#	res = Result()
#	res.model = model
#	for i in range(len(evalSOP.testNames)):
#		perf = getPerformanceLabel(evalSOP.testNames[i])
#		session.add(perf)
#		res.performances.append(ResultPerf(perf, evalSOP.meanPerf[i]))
#	session.add(res)
#	session.commit()
#	closeSession(session)
	SOPPerf = dict()
	for performance in evalSOP.testNames: 
		SOPPerf[(performance, modelSOP.alpha, modelSOP.teta)] = evalSOP.meanPerf[[i for i,j in enumerate(evalSOP.testNames) if j==performance]][0]
	print 'Writing Baskets to file'
	file = open(resultFolder+'SOPPerf_a%s_t%s_nb%s_nr%s.txt'%(alpha,teta,nbBasket,nbReco),'w')
	pickle.dump(SOPPerf, file)
	file.close()  
	#evalSOP.savePerf(resultFolder+'SOPPerf_a%s_t%s_nb%s_nr%s.txt'%(alpha,teta,nbBasket,nbReco))
	return evalSOP

def Cosine(nbBasket, nbReco):
	data = load()
	###############################################################
	# CREATE MODELS
	###############################################################
	print 'Create the model based on the training set'
	
	modelCosine = processing.CosineRecoModel(data.getUserItemMatrix())
		
	###############################################################
	# SET RECOMMENDATION
	###############################################################
	if nbBasket == -1:
		evalCosine = processing.Evaluation(modelCosine, data.getBasketItemList(), nbReco)
	else :
		evalCosine = processing.Evaluation(modelCosine, data.getBasketItemList()[:nbBasket], nbReco)
	
	###############################################################
	# LAUNCH RECOMMENDATION + SAVE RESULTS
	###############################################################	
	t = time.time()
	evalCosine.newEval()
	CosineTime = time.time()-t
	mmwrite(resultFolder+'Cosine_nb%s'%nbBasket,evalCosine.perf) 
	
	CosinePerf = dict()
	for performance in evalCosine.testNames: 
		CosinePerf[performance] = evalCosine.meanPerf[[i for i,j in enumerate(evalCosine.testNames) if j==performance]][0]
	print 'Writing Baskets to file'
	file = open(resultFolder+'CosinePerf_nb%s_nr%s.txt'%(nbBasket,nbReco),'w')
	pickle.dump(CosinePerf, file)
	file.close()  
	
	print 'Cosine Execution time:', CosineTime
	print 'Performances :'
	print evalCosine.testNames
	print evalCosine.computePerf()
	evalCosine.savePerf(resultFolder+'Cosine_nb%s.txt'%nbBasket)
	return evalCosine

def CondProb(alpha, nbBasket, nbReco):
	data = load()
	###############################################################
	# CREATE MODELS
	###############################################################
	print 'Create the model based on the training set'
	
	modelCondProb = processing.CondProbRecoModel(data.getUserItemMatrix(), alpha)
		
	###############################################################
	# SET RECOMMENDATION
	###############################################################
	if nbBasket == -1:
		evalCondProb = processing.Evaluation(modelCondProb, data.getBasketItemList(), nbReco)
	else :
		evalCondProb = processing.Evaluation(modelCondProb, data.getBasketItemList()[:nbBasket], nbReco)
	
	###############################################################
	# LAUNCH RECOMMENDATION + SAVE RESULTS
	###############################################################	
	t = time.time()
	evalCondProb.newEval()
	CondProbTime = time.time()-t
	mmwrite(resultFolder+'CondProb_a%s_nb%s'%(alpha, nbBasket),evalCondProb.perf) 
	
	print 'Cond Prob Execution time:', CondProbTime
	print 'Performances :'
	print evalCondProb.testNames
	print evalCondProb.computePerf()
	evalCondProb.savePerf(resultFolder+'CondProb_a%s_nb%s.txt'%(alpha, nbBasket))
	return evalCondProb

def RW(alpha, nbBasket, nbReco):
	data = load()
	###############################################################
	# CREATE MODELS
	###############################################################
	print 'Create the model based on the training set'
	
	modelRW = processing.BasketRandomWalk(data.getUserItemMatrix(), alpha)
		
	###############################################################
	# SET RECOMMENDATION
	###############################################################
	if nbBasket == -1:
		evalRW = processing.Evaluation(modelRW, data.getBasketItemList(), nbReco)
	else :
		evalRW = processing.Evaluation(modelRW, data.getBasketItemList()[:nbBasket], nbReco)
	
	###############################################################
	# LAUNCH RECOMMENDATION + SAVE RESULTS
	###############################################################	
	t = time.time()
	evalRW.newEval()
	RWTime = time.time()-t
	mmwrite(resultFolder+'RW_a%s_nb%s'%(alpha, nbBasket),evalRW.perf) 
	
	print 'RW Execution time:', RWTime
	print 'Performances :'
	print evalRW.testNames
	print evalRW.computePerf()
	evalRW.savePerf(resultFolder+'RW_a%s_nb%s.txt'%(alpha, nbBasket))
	return evalRW

def RW_POP(alpha, nbBasket, nbReco):
	data = load()
	###############################################################
	# CREATE MODELS
	###############################################################
	print 'Create the model based on the training set'
	
	modelRW = processing.BasketRandomWalk_POP(data.getUserItemMatrix(), alpha)
		
	###############################################################
	# SET RECOMMENDATION
	###############################################################
	if nbBasket == -1:
		evalRW = processing.Evaluation(modelRW, data.getBasketItemList(), nbReco)
	else :
		evalRW = processing.Evaluation(modelRW, data.getBasketItemList()[:nbBasket], nbReco)
	
	###############################################################
	# LAUNCH RECOMMENDATION + SAVE RESULTS
	###############################################################	
	t = time.time()
	evalRW.newEval()
	RWTime = time.time()-t
	mmwrite(resultFolder+'RW_POP_a%s_nb%s'%(alpha, nbBasket),evalRW.perf) 
	
	print 'RW_POP Execution time:', RWTime
	print 'Performances :'
	print evalRW.testNames
	print evalRW.computePerf()
	evalRW.savePerf(resultFolder+'RW_POP_a%s_nb%s.txt'%(alpha, nbBasket))
	return evalRW

def RWWR(alpha, sim, nbBasket, nbReco):
	
	data = load()
	#data = cvTaFengProcessing()
	###############################################################
	# CREATE MODELS
	###############################################################
	print 'Create the model based on the training set'
	
	modelRWWR = processing.RandomWalkWithRestartRecoModel(data.getUserItemMatrix(), alpha, sim)
	modelRWWR.train()
	###############################################################
	# SET RECOMMENDATION
	###############################################################
	if nbBasket == -1:
		evalRWWR = processing.Evaluation(modelRWWR, data.getBasketItemList(), nbReco)
	else :
		evalRWWR = processing.Evaluation(modelRWWR, data.getBasketItemList()[:nbBasket], nbReco)
	
	###############################################################
	# LAUNCH RECOMMENDATION + SAVE RESULTS
	###############################################################	
	t = time.time()
	evalRWWR.newEval()
	RWWRTime = time.time()-t
	mmwrite(resultFolder+'RWWR_a%s_nb%s'%(alpha, nbBasket),evalRWWR.perf) 
	
	print 'RWWR Execution time:', RWWRTime
	print 'Performances :'
	print evalRWWR.testNames
	print evalRWWR.computePerf()
	
	RWWRPerf = dict()
	for performance in evalRWWR.testNames: 
		RWWRPerf[(performance, modelRWWR.alpha, modelRWWR.sim)] = evalRWWR.meanPerf[[i for i,j in enumerate(evalRWWR.testNames) if j==performance]][0]
	print 'Writing Baskets to file'
	file = open(resultFolder+'RWWRPerf_a%s_sim%s_nb%s_nr%s.txt'%(alpha,sim, nbBasket,nbReco),'w')
	pickle.dump(RWWRPerf, file)
	file.close() 
	
	#evalRWWR.savePerf(resultFolder+'RWWR_a%s_nb%s'%(alpha, nbBasket))
	return evalRWWR

def BRWWR(alpha, theta, nbBasket, nbReco):
	
	data = load()
	#data = cvTaFengProcessing()
	###############################################################
	# CREATE MODELS
	###############################################################
	print 'Create the model based on the training set'
	
	modelBRWWR = processing.BiasedRandomWalkWithRestartRecoModel(data.getUserItemMatrix(), theta, alpha)
	modelBRWWR.train()
	###############################################################
	# SET RECOMMENDATION
	###############################################################
	if nbBasket == -1:
		evalBRWWR = processing.Evaluation(modelBRWWR, data.getBasketItemList(), nbReco)
	else :
		evalBRWWR = processing.Evaluation(modelBRWWR, data.getBasketItemList()[:nbBasket], nbReco)
	
	###############################################################
	# LAUNCH RECOMMENDATION + SAVE RESULTS
	###############################################################	
	t = time.time()
	evalBRWWR.newEval()
	BRWWRTime = time.time()-t
	mmwrite(resultFolder+'BRWWR_a%s_nb%s'%(alpha, nbBasket),evalBRWWR.perf) 
	
	print 'BRWWR Execution time:', BRWWRTime
	print 'Performances :'
	print evalBRWWR.testNames
	print evalBRWWR.computePerf()
	
	BRWWRPerf = dict()
	for performance in evalBRWWR.testNames: 
		BRWWRPerf[(performance, modelBRWWR.alpha)] = evalBRWWR.meanPerf[[i for i,j in enumerate(evalBRWWR.testNames) if j==performance]][0]
	print 'Writing Baskets to file'
	file = open(resultFolder+'BRWWRPerf_a%s_sim%s_nb%s_nr%s.txt'%(alpha, nbBasket,nbReco),'w')
	pickle.dump(BRWWRPerf, file)
	file.close() 
	
	#evalRWWR.savePerf(resultFolder+'RWWR_a%s_nb%s'%(alpha, nbBasket))
	return evalBRWWR

def pLSA(Z, nbBasket, nbReco):
	data = load()
	###############################################################
	# CREATE MODELS
	###############################################################
	print 'Create the model based on the training set'
	
	if not os.path.exists('pLSAModel_Z%s.dat'%Z):
		modelpLSA = processing.PLSARecoModel(data.getUserItemMatrix(), Z)
		modelpLSA.train()
	
		file = open('pLSAModel_Z%s.dat'%Z, 'w')
		pickle.dump(modelpLSA.pLSAmodel.get_model(), file)
		file.close()
	else :
		modelpLSA = processing.PLSARecoModel(data.getUserItemMatrix(), Z)
		file = open('pLSAModel_Z%s.dat'%Z, 'r')
		modelpLSA.pLSAmodel.set_model(pickle.load(file))
		file.close()

	###############################################################
	# SET RECOMMENDATION
	###############################################################
	if nbBasket == -1:
		evalpLSA = processing.Evaluation(modelpLSA, data.getBasketItemList(), nbReco)
	else :
		evalpLSA = processing.Evaluation(modelpLSA, data.getBasketItemList()[:nbBasket], nbReco)
	
	###############################################################
	# LAUNCH RECOMMENDATION + SAVE RESULTS
	###############################################################	
	t = time.time()
	evalpLSA.newEval()
	pLSATime = time.time()-t
	mmwrite(resultFolder+'PLSA_Z%s_nb%s'%(Z, nbBasket),evalpLSA.perf) 
	
	print 'pLSA Execution time:', pLSATime
	print 'Performances :'
	print evalpLSA.testNames
	print evalpLSA.computePerf()
	
	pLSAPerf = dict()
	for performance in evalpLSA.testNames: 
		pLSAPerf[(performance, modelpLSA.Z)] = evalpLSA.meanPerf[[i for i,j in enumerate(evalpLSA.testNames) if j==performance]][0]
	print 'Writing Baskets to file'
	file = open(resultFolder+'pLSAPerf_Z%s_nb%s_nr%s.txt'%(Z, nbBasket,nbReco),'w')
	pickle.dump(pLSAPerf, file)
	file.close() 
	
	#evalpLSA.savePerf(resultFolder+'pLSA_Z%s_nb%s'%(Z, nbBasket))
	return evalpLSA


def Pop(nbBasket, nbReco):
	data = load()
	###############################################################
	# CREATE MODELS
	###############################################################
	print 'Create the model based on the training set'
	
	modelPop = processing.PopRecoModel(data.getUserItemMatrix())
		
	###############################################################
	# SET RECOMMENDATION
	###############################################################
	if nbBasket == -1:
		evalPop = processing.Evaluation(modelPop, data.getBasketItemList(), nbReco)
	else :
		evalPop = processing.Evaluation(modelPop, data.getBasketItemList()[:nbBasket], nbReco)
	
	###############################################################
	# LAUNCH RECOMMENDATION + SAVE RESULTS
	###############################################################	
	t = time.time()
	evalPop.newEval()
	PopTime = time.time()-t
	mmwrite(resultFolder+'Pop',evalPop.perf) 
	
	print 'Pop Execution time:', PopTime
	print 'Performances :'
	print evalPop.testNames
	print evalPop.computePerf()
	evalPop.savePerf(resultFolder+'Pop.txt')
	return evalPop




data = load()
baseProcessing = processing.RecoModel(data.getUserItemMatrix())

def Multi_BRWWR():
	
	alpha_theta = [(0.9, 0), (0.8, 0), (0.5, 0), (0.2, 0), (0.1, 0),
					(0.9, 1), (0.8, 1), (0.5, 1), (0.2, 1), (0.1, 1),
					(0.9, 0.1), (0.8,0.1), (0.5,0.1), (0.2,0.1), (0.1, 0.1),
					(0.9, 0.01), (0.8,0.01), (0.5,0.01), (0.2,0.01), (0.1, 0.01)]
	nbBasket = 5000
	sim = 'cos'
	nbReco = 3

	for alpha, theta in alpha_theta:
		print '###############################################################'
		print '# Start Recommendation with alpha : %s,  and theta : %s'%(alpha, theta)
		print '###############################################################'
		modelBRWWR = processing.BiasedRandomWalkWithRestartRecoModel(baseProcessing, theta, alpha, sim)	
		modelBRWWR.train()
		###############################################################
		# SET RECOMMENDATION
		###############################################################
		if nbBasket == -1:
			evalBRWWR = processing.Evaluation(modelBRWWR, data.getBasketItemList(), nbReco)
		else :
			evalBRWWR = processing.Evaluation(modelBRWWR, data.getBasketItemList()[:nbBasket], nbReco)
		
		###############################################################
		# LAUNCH RECOMMENDATION + SAVE RESULTS
		###############################################################	
		t = time.time()
		evalBRWWR.newEval()
		BRWWRTime = time.time()-t
		mmwrite(resultFolder+'BRWWR_a%s_t%s_nb%s'%(alpha, theta, nbBasket),evalBRWWR.perf) 
		
		print 'BRWWR Execution time:', BRWWRTime
		print 'Performances :'
		print evalBRWWR.testNames
		print evalBRWWR.computePerf()
		
		BRWWRPerf = dict()
		for performance in evalBRWWR.testNames: 
			BRWWRPerf[(performance, modelBRWWR.alpha, modelBRWWR.theta)] = evalBRWWR.meanPerf[[i for i,j in enumerate(evalBRWWR.testNames) if j==performance]][0]
		print 'Writing Baskets to file'
		file = open(resultFolder+'BRWWRPerf_a%s_t%s_nb%s_nr%s.txt'%(alpha,theta, nbBasket,nbReco),'w')
		pickle.dump(BRWWRPerf, file)
		file.close() 
	

def Multi_RW():
	a = [0.01, 0.05, 0.1, 0.5, 0.8, 0.9]
	#a= [0.9]
	sim = 'cos'
	nbBasket = 10000
	nbReco = 3
	
	for alpha in a:
		print '###############################################################'
		print '# Start Recommendation with alpha : %s'%alpha
		print '###############################################################'
		modelRWWR = processing.RandomWalkWithRestartRecoModel(baseProcessing, alpha, sim)	
		modelRWWR.train()
		###############################################################
		# SET RECOMMENDATION
		###############################################################
		if nbBasket == -1:
			evalRWWR = processing.Evaluation(modelRWWR, data.getBasketItemList(), nbReco)
		else :
			evalRWWR = processing.Evaluation(modelRWWR, data.getBasketItemList()[:nbBasket], nbReco)
		
		###############################################################
		# LAUNCH RECOMMENDATION + SAVE RESULTS
		###############################################################	
		t = time.time()
		evalRWWR.newEval()
		RWWRTime = time.time()-t
		mmwrite(resultFolder+'RWWR_a%s_nb%s'%(alpha, nbBasket),evalRWWR.perf) 
		
		print 'RWWR Execution time:', RWWRTime
		print 'Performances :'
		print evalRWWR.testNames
		print evalRWWR.computePerf()
		
		RWWRPerf = dict()
		for performance in evalRWWR.testNames: 
			RWWRPerf[(performance, modelRWWR.alpha, modelRWWR.sim)] = evalRWWR.meanPerf[[i for i,j in enumerate(evalRWWR.testNames) if j==performance]][0]
		print 'Writing Baskets to file'
		file = open(resultFolder+'RWWRPerf_a%s_sim%s_nb%s_nr%s.txt'%(alpha,sim, nbBasket,nbReco),'w')
		pickle.dump(RWWRPerf, file)
		file.close() 
	
def Multi_Cosine():
	nbBasket = 10000
	nbReco = 3
	modelCosine = processing.CosineRecoModel(baseProcessing)

	###############################################################
	# SET RECOMMENDATION
	###############################################################
	if nbBasket == -1:
		evalCosine = processing.Evaluation(modelCosine, data.getBasketItemList(), nbReco)
	else :
		evalCosine = processing.Evaluation(modelCosine, data.getBasketItemList()[:nbBasket], nbReco)	
	###############################################################
	# LAUNCH RECOMMENDATION + SAVE RESULTS
	###############################################################	
	t = time.time()
	evalCosine.newEval()
	CosineTime = time.time()-t
	mmwrite(resultFolder+'Cosine_nb%s'%nbBasket,evalCosine.perf) 
	
	CosinePerf = dict()
	for performance in evalCosine.testNames: 
		CosinePerf[performance] = evalCosine.meanPerf[[i for i,j in enumerate(evalCosine.testNames) if j==performance]][0]
	print 'Writing Baskets to file'
	fileToWrite = open(resultFolder+'CosinePerf_nb%s_nr%s.txt'%(nbBasket,nbReco),'w')
	pickle.dump(CosinePerf, fileToWrite)
	fileToWrite.close()  
	
	print 'Cosine Execution time:', CosineTime
	print 'Performances :'
	print evalCosine.testNames
	print evalCosine.computePerf()
	evalCosine.savePerf(resultFolder+'Cosine_nb%s.txt'%nbBasket)


def drawSomeThing(model, nb='nb5000'):

	expBRWWR = {}
	expRWWR = {}
	expCosine = {}

	for dirname, dirnames, filenames in os.walk('../result'):
		#for subdirname in dirnames:
		#    print os.path.join(dirname, subdirname)
		for filename in filenames:
			filePath = os.path.join(dirname, filename)
			if '/BRWWRPerf' in filePath and nb in filePath:
				print filePath
				tmpDict = pickle.load(open(filePath, 'r'))

				for perf, a, t in tmpDict:
					if perf in expBRWWR:
						if a in expBRWWR[perf]:
							expBRWWR[perf][a][t] = tmpDict[(perf, a, t)]
						else :
							expBRWWR[perf][a] = {t : tmpDict[(perf, a, t)]}
					else :
						expBRWWR[perf] = {a:{t:tmpDict[(perf, a, t)]}}

			if '/RWWRPerf' in filePath and nb in filePath:
				print filePath
				tmpDict = pickle.load(open(filePath, 'r'))

				for perf, a, sim in tmpDict:
					if perf in expRWWR:
						if a in expRWWR[perf]:
							expRWWR[perf][a][sim] = tmpDict[(perf, a, sim)]
						else :
							expRWWR[perf][a] = {sim : tmpDict[(perf, a, sim)]}
					else :
						expRWWR[perf] = {a:{sim:tmpDict[(perf, a, sim)]}}

			if '/CosinePerf' in filePath and nb in filePath:
				print filePath
				tmpDict = pickle.load(open(filePath, 'r'))
				for perf in tmpDict:
					expCosine[perf] = tmpDict[perf]


	plt.figure()
	for n, forPerf in enumerate(['_perf_Novelty', '_perf_bHR_pop', '_perf_wHR', '_perf_bHR_rnd']):
		plt.subplot(2,2,n)
		for j in sort([t for t in expBRWWR[forPerf][0.1]]):
			xData = []
			yData = []
			for i in sort([a for a in expBRWWR[forPerf]]):
				xData.append(i)
				yData.append(expBRWWR[forPerf][i][j])
			plt.title("BRWWR for %s"%forPerf)
			plt.xlabel("alpha")
			plt.ylabel(forPerf)
			plt.plot(xData, yData, label='t:%s'%(j))
		#Baseline
		plt.plot(xData, [expCosine[forPerf]]*len(xData), label='Cosine')
		plt.legend()
	plt.show()


	plt.figure()
	for n, forPerf in enumerate(['_perf_Novelty', '_perf_bHR_pop', '_perf_wHR', '_perf_bHR_rnd']):
		plt.subplot(2,2,n)
		for j in sort([sim for sim in expRWWR[forPerf][0.1]]):
			xData = []
			yData = []
			for i in sort([a for a in expRWWR[forPerf]]):
				xData.append(i)
				yData.append(expRWWR[forPerf][i][j])
			plt.title("RWWR for %s"%forPerf)
			plt.xlabel("alpha")
			plt.ylabel(forPerf)
			plt.plot(xData, yData, label='Sim:%s'%(j))
		#Baseline
		plt.plot(xData, [expCosine[forPerf]]*len(xData), label='Cosine')
		plt.legend()
	plt.show()




