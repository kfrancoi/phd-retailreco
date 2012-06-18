#!/usr/bin/env python
# encoding: utf-8
"""
untitled.py

Created by Kevin Fran�oisse on 2010-03-31.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.
"""

import os
import sys
import pickle
import scipy.sparse as sparse
import scipy.sparse.linalg as spLinalg
from scipy.io import mmwrite, mmread
from numpy import *
import random
import distance
import toolBox
import time
import getopt


class BasketPreprocessing:
	
	def __init__(self):
		
		self.userDict = {}
		self.itemDict = {}
		self.basketDict = {}
		self.saveUI = 'sUI.mtx'
		self.saveItemDict = 'itemDict.dict'
		self.saveBI = 'basketDict.dict'
	
	def trainDataPreprocessing(self, dataPaths, saveUI='sUI', saveItemDict='itemDict.dict'):
		
		if saveUI != None:
			self.saveUI = saveUI
		
		if  os.path.exists(self.saveUI):
			self.loadTrainData()
		else :
			print 'Start building weight matrix...'
			
			ind_u = 0
			ind_i = 0
			for path in dataPaths:
				print 'Parsing data %s'%path
				data = loadtxt(path, delimiter=';', skiprows=1, usecols = (1,4))
				for u,i in data:
					if int(u) not in self.userDict.keys():
						self.userDict[int(u)] = ind_u
						ind_u+=1
					if i not in self.itemDict.keys():
						self.itemDict[i] = ind_i
						ind_i+=1
			
			
			self.UI = zeros((len(self.userDict.keys()), len(self.itemDict.keys())))
			
			# Population of the transaction matrix UI
			# ---------------------------------------
			for path in dataPaths:
				data = loadtxt(path, delimiter=';', skiprows=1, usecols = (1,4))
				print 'Recording transactions...'
				for u,i in data:
					self.UI[self.userDict[int(u)], self.itemDict[i]]+=1
		
			# Log frequence UI transformation (TO BE OPTIMIZE)
			# -------------------------------
			print 'Frequence normalisation'
			for i,j in argwhere(self.UI!=0):
				self.UI[i,j] = log(self.UI[i,j]+1)
		
	def testDataPreprocessing(self, path, saveBI='basketDict.dict'):
		
		if saveBI != None:
			self.saveBI = saveBI
		if self.itemDict == None:
			if os.path.exists(self.saveItemDict):
				self.loadTrainData()
			else :
				print 'You must first preprocess your training data'
				raise ValueError

		print 'Preprocess test baskets'
		event_fields = ['datetime', 'customerID', 'item']
		event_format = ['S20', 'int', 'int']
		dt=dtype({'names': [x for x in event_fields], 'formats': [b for b in event_format]})
		data = loadtxt(path, delimiter=';', skiprows=1, usecols = (0,1,4), dtype=dt) #February
		
		#Create (date, customer) key for the dictionary
		print 'Create test baskets dictionnary'
		ind_b = 0
		for i in arange(len(data)):
			if (data[i][0],data[i][1]) not in self.basketDict.keys():
				self.basketDict[(data[i][0],data[i][1])] = ind_b
				ind_b+=1
		
		print 'Create basket array'
		self.BasketItem = zeros((len(self.basketDict), len(self.itemDict.keys())), dtype='int')
		for i in arange(len(data)):
			if data[i][2] in self.itemDict.keys():
				self.BasketItem[self.basketDict[(data[i][0], data[i][1])], self.itemDict[data[i][2]]] +=1
		
		
		self.BasketItemList = []
		for i in xrange(self.BasketItem.shape[0]):
			self.BasketItemList.append(argwhere(self.BasketItem[i,:]!=0).flatten())
	
	def saveTrainData(self):
		
		print 'Writing UI to file'
		sUI = sparse.csr_matrix(self.UI)
		mmwrite(self.saveUI, sUI)
		
		print 'Writing itemDict to file'
		file = open(self.saveItemDict, 'w') # write mode
		pickle.dump(self.itemDict, file)
		file.close()

	def saveBasketData(self):
		print 'Writing Baskets to file'
		file = open(self.saveBI, 'w')
		pickle.dump(self.BasketItemList, file)
		file.close()
		
	def loadTrainData(self):
		print 'Read data from file'
		sUI = mmread(self.saveUI)
		self.UI = sUI.toarray()
		
		print 'Read itemDict from file'
		file = open(self.saveItemDict, 'r')
		self.itemDict = pickle.load(file)
		file.close()
		
	def loadBasketData(self):
		print 'Read BI from file'
		file = open(self.saveBI, 'r')
		self.BasketItemList = pickle.load(file)
		file.close()
	
	def getUserItemMatrix(self):
		return self.UI
	
	def getBasketItemList(self):
		return self.BasketItemList

class RecoModel(object):
	
	def __init__(self, UI):
		
		self.computePior(UI)
		
	def computePior(self, UI):
	
		#print 'Compute Item popularity'
		#tempUI = zeros(self.UI.shape)
		#tempUI[where(self.UI!=0)] = 1
		#s = sum(tempUI, 0)
		
		print 'Compute Item frequency'
		s = sum(self.UI,0)
		#self.ItemPop = (s - mean(s))*1.0 / std(s)
		self.ItemPop = s
	
		print 'Compute Item prior probability'
		self.ItemPrior = self.ItemPop / sum(self.ItemPop)
		
		self.nbUsers = self.UI.shape[0]
		self.nbItems = self.UI.shape[1]

class BasketRandomWalk(RecoModel):
	'''
	This class build a basket sensitive random walk model on bipartite network
	'''
	def __init__(self, UI, d=0.8):
		self.UI = UI
		self.d = d
		super(BasketRandomWalk, self).__init__(self.UI)
		self._buildRandomWalkScore(d)
	
	def __repr__(self):
		return '<BaskerRandomWalk: alpha %s>'%self.d
	
	def _buildRandomWalkScore(self,d):
		
		epsilon = 0.001
		#P = matrix(toolBox.randomWalkTransitionProbability(self.UI))
		P = matrix(toolBox.AtoP(toolBox.cosineSimilarity(self.UI)))
		R = matrix(zeros((self.nbItems, self.nbItems)), dtype=float64)
		U = matrix(eye(self.nbItems, self.nbItems,dtype=float64))
		#U = U / self.nbItems
		diff = 1000
		count = 20
		print 'Convergence ...'
		while (diff > epsilon) and (count > 0):
			#print P,R,U
			oldRSum= sum(R)
			R = d*P*R + (1-d)*U
			diff = abs(oldRSum - sum(R)) / sum(R)
			count-=1
			print diff
		
		#for i in arange(R.shape[0]):
		#	R[i,i] = 1
		print R
		print sum(R,0)
		self.R = array(R)
	
	def recommend(self, evidences, n):
		self.scores = sum(self.R[evidences, :],0)
		self.scores[evidences] = 0
		
		ind = argsort(self.scores)
		
		return ind[-n:] #Return the n largest scores

class CondProbRecoModel(RecoModel):
	'''
	This class build a conditional probability similarity model and contains all information
	to make recommendation
	'''
	def __init__(self, UI, alpha=0.5):
		self.UI = UI
		self.alpha = alpha
		super(CondProbRecoModel, self).__init__(UI)
		self._buildSimMatrix()
	
	def __repr__(self):
		return '<CondProbRecoModel: alpha %s>'%self.alpha
	
	def _buildSimMatrix(self):
		self.Sim = toolBox.condProbSimilarity(self.UI, self.alpha)
	
	def recommend(self, evidences, n):
		self.scores = sum(self.Sim[evidences, :],0)
		self.scores[evidences] = 0
		
		ind = argsort(self.scores)
		
		return ind[-n:] #Return the n largest scores

class CosineRecoModel(RecoModel):
	'''
	This class build a cosine and contain all information to make recommendation
	'''
	def __init__(self, UI):
		self.UI = UI
		super(CosineRecoModel,self).__init__(UI)
		self._buildSimMatrix()
	
	def __repr__(self):
		return '<CosineRecoModel>'
	
	def _buildSimMatrix(self):
		self.Sim = array(toolBox.cosineSimilarity(self.UI))
	
	def recommend(self, evidences, n):
		self.scores = sum(self.Sim[evidences, :],0)
		self.scores[evidences] = 0
		
		ind = argsort(self.scores)
		
		if n == -1:
			return ind
		else:
			return ind[-n:] #Return the n largest scores
	
	def getScores(self, evidences, n):
		self.recommend(evidences, n)
		return self.scores

class SOPRecoModel(RecoModel):
	'''
	This class contain all model information on trained data
	'''
	def __init__(self, UI, alpha=0.5, teta=0.9):
		'''
		Init method
		@param UI: User-Item matrix
		'''
		self.UI = UI #Log freq ???
		self.alpha = alpha
		self.teta = teta
		super(SOPRecoModel,self).__init__(UI)
	
	def __repr__(self):
		return '<SOPRecoModel: alpha %s teta %s>'%(self.alpha,self.teta)
	
	def computeSim(self, type='cosine'):
		
		print 'Compute similarity'
		self.Sim = toolBox.cosineSimilarity(self.UI)
	
	def computeCostMatrix(self):
		
		print 'Compute cost matrix'
		C = toolBox.sqrtEuclideanDistance(self.Sim)
		C = C/std(C)
		
		#In degree : number of user having bought product
		temp = zeros(self.UI.shape, dtype='int32')
		for i,j in argwhere(self.UI!=0):
			temp[i,j] = 1
		Phi = sum(temp, 0) #-->0.06
		#Phi = self.ItemPrior #-->0.08
		#Phi = Phi / linalg.norm(Phi)
		Phi = log(1+Phi)
		Phi = Phi / std(Phi)
		mPhi = repeat(array([Phi]), Phi.shape[0], axis=0)
		
		Ctot = self.alpha * C + (1-self.alpha) * mPhi
		
		return Ctot
	
	def launch(self):
		#Split this methods into two parts to be able to quickly output Weight matrix for different values of alpha and teta
		#Cosine computation
		self.computeSim('cosine')
		# Removing Similarity between two item
		#for i in range(self.Sim.shape[0]):
		#	self.Sim[i,i]=0
		#Compute cost matrix
		self.Ctot = array(self.computeCostMatrix())
		#Compute afinity matrix
		self.P = array(toolBox.AtoP(self.Sim))
		#Compute weight matrix
		self.W = exp(-self.teta*self.Ctot) * self.P
	
	def recommend(self, evidences, n):
		
		print 'Recommend items'	
		We, h0, hf = self.dummyExtension(evidences)
		We = matrix(We)
		h0 = matrix(h0)
		hf = matrix(hf)
		#Solve equation systems
		#TODO : This need to be done in Cython !
		row, col = We.shape
		I = eye(row,col)
		try :
			
			zf = spLinalg.cgs(I-We.T,h0)
			#zf = linalg.solve(I-We.T,h0)
			zb = spLinalg.cgs(I-We,hf)
			#zb = linalg.solve(I-We,hf)
		except linalg.LinAlgError:
			print 'Alpha %s and teta %s produce a Singular matrix exception'%(self.alpha,self.teta)
			raise linalg.LinAlgError
		
		zb = matrix(zb[0].reshape(zb[0].shape[0],1))
		zf = matrix(zf[0].reshape(zf[0].shape[0],1))
		
		Z = h0.T * zb - h0.T * hf
		
		zf = array(zf)
		zb = array(zb)
		h0 = array(h0)
		Z = array(Z)
		b = ((zf - h0) * zb) / Z
		
		b[evidences] = 0
		bInd = argsort(b[:,0])
		
		#Scores
		self.b = b.flatten()
		
		if n == -1:
			return bInd
		else:
			return bInd[-n:]
	
	def getScores(self, evidences, n):
		self.recommend(evidences, n)
		return self.b
		
	
	def dummyExtension(self, nodeList):
		'''
		Create new dummy node that link out node of interest node - nodeList.
		Weight matrix (W) is augmented
		'''
		extension = self.W[nodeList,:]
		temp = concatenate((self.W,extension),0)
		Xextended = concatenate((temp, zeros((temp.shape[0],extension.shape[0]))),1)
		
		hf = zeros((Xextended.shape[0],1))
		hf[nodeList] = 1
		
		h0 = zeros((Xextended.shape[0],1))
		h0[-len(nodeList):] = 1
		
		#Remove out link from real basket nodes
		Xextended[nodeList,:] = 0
		
		return Xextended, h0, hf

class RandomWalkWithRestartRecoModel(RecoModel):
	
	def __init__(self, UI, alpha):
		self.UI = UI
		self.alpha = alpha
		super(RandomWalkWithRestartRecoModel, self).__init__(self.UI)
		self._buildMatrices()
	
	def __repr__(self):
		return '<RandomWalkWithRestartRecoModel: alpha %s>'%self.alpha
	
	def _buildMatrices(self):
		'''Different ways of computing the transition probability matrix can be considered'''
		print 'Compute Sim & P'
		self.Sim = toolBox.cosineSimilarity(self.UI)
		self.P = toolBox.AtoP(self.Sim)
	
	def recommend(self, evidences, n):
		
		I = eye(self.nbItems)
		u = zeros(self.nbItems)
		u[evidences] = 1 #1/nbEvidences ???
		try:
			r = spLinalg.cgs(I-(self.alpha * self.P),(1-self.alpha)*u)
		except linalg.LinAlgError:
			print 'Linear Algebra Error'
			raise linalg.LinAlgError
		
		r[0][evidences] = 0
		ind = argsort(r[0]) #Item with highest scores are recommended
		
		return ind[-n:]
		

class PopRecoModel(RecoModel):
	
	def __init__(self,UI):
		self.UI = UI
		super(PopRecoModel, self).__init__(UI)
	
	def __repr__(self):
		return '<PopRecoModel>'
	
	def recommend(self, evidences, n):
		ind = argsort(self.ItemPop)
		ind = [i for i in ind if i not in evidences]
		
		return ind[-n:]

class Evaluation:
	'''
	Evaluation protocol of the recommender system
	'''
	def __init__(self, model, testBasketsList, n):
		'''Class constructor
		@type model : RecoModel
		@param model: recommendation model object. Object must give functions:
		- recommend(evidence)
		- ItemPop
		- ItemPrior
		As function recommend(evidence)
		@type evidenceBaskets: List of numpy array
		@param evidenceBaskets: List of numpy array of basket items 
		@type n: integer
		@param n: Number of item the recommender model must return to assess performances
		'''
		
		self.model = model
		self.evidenceBaskets = testBasketsList
		self.nbBaskets = len(self.evidenceBaskets)
		self.n = n
		self.removedList=[] # List of basket with less than 4 items
		
		#Create list of existing test
		self.testNames= []			
		for member in dir(self):
			if member[0:6] == '_perf_': #A test method must begin with _perf_
				self.testNames.append(member)
			
	def newEval(self):
		
		self.perf = zeros((self.nbBaskets, len(self.testNames)))
		for basket in arange(len(self.evidenceBaskets)):
			print 'Basket number %s'%basket
			if len(self.evidenceBaskets[basket]) >=4 :
				for test in arange(len(self.testNames)):
					#print '---> Performance %s'%self.testNames[test]
					self.perf[basket,test] = getattr(self, self.testNames[test])(self.evidenceBaskets[basket])
			else:
				self.removedList.append(basket)
	
	def meanPerf(self):
		
		self.meanPerf = zeros(self.perf.shape[1], dtype='float')
		self.meanPerf[0] = mean(self.perf[:,0])
		self.meanPerf[1] = mean(self.perf[:,1])
		self.meanPerf[2] = mean(self.perf[:,2])
		return self.meanPerf
	
	def savePerf(self, fname):
		
		file = open(fname, 'w')
		file.write('%s\n'%self.model)
		file.write('%s\n'%self.testNames)
		file.write('%s\n'%self.meanPerf)
		file.close()
		
	def _perf_bHR_pop(self, evidenceBasket):
		'''
		binary hit rate using x least popular item
		@param evidenceBasket: ItemPop test item representing a transaction 
		'''
		#print 'EvidenceBasket : %s'%evidenceBasket
		popInd = argsort(self.model.ItemPop[evidenceBasket])
		sortedEvidenceBasket = evidenceBasket[popInd] # TO FIX !!!!
		targets = sortedEvidenceBasket[:self.n] #Define targets according to pop
		#targets = sortedEvidenceBasket[-self.n:]
		evidences = [item for item in evidenceBasket if item not in targets] #Define evidences
		recoItems = self.model.recommend(evidences, self.n) #Recommend
		
		print 'Evidences : %s'%evidences
		print 'RecoItem : %s'%recoItems
		print 'Targets : %s'%targets
		
		for item in recoItems:
			if item in targets:
				return 1
		return 0
		#return sum([1 for item in recoItems if item in targets])
		#return sum([1 for item in recoItems if item in targets]) #TO CHECK : May be only 1 or 0
	
	def _perf_bHR_rnd(self, evidenceBasket):
		'''
		binary hit rate using a random leave 3 out protocol
		@param evidenceBasket: List of test item representing a transaction
		'''
		
		random.shuffle(evidenceBasket)
		targets = evidenceBasket[:self.n]
		evidences = evidenceBasket[self.n:]
		recoItems = self.model.recommend(evidences, self.n)
		
		print 'RecoItem : %s'%recoItems
		print 'Evidences : %s'%evidences
		print 'Targets : %s'%targets
		
		#return sum([1 for item in recoItems if item in targets]) #TO CHECK : May be only 1 or 0
		for item in recoItems:
			if item in targets:
				return 1
		return 0
		
	def _perf_wHR(self, evidenceBasket):
		'''
		weighted hit rate with leave one out cross validation protocol
		@param evidenceBasket: List of test item representing a transaction
		'''
		#loo protocol
		hits = zeros(len(evidenceBasket), dtype='int32')
		for i in range(len(evidenceBasket)):
			evidences = [item for item in evidenceBasket if item!=evidenceBasket[i]] #Tout evidence except for item
			target = evidenceBasket[i]
			recoItems = self.model.recommend(evidences, self.n)
			#print 'RecoItem : %s'%recoItems
			#print 'Evidences : %s'%evidences
			#print 'Targets : %s'%target
			if target in recoItems:
				hits[i] = 1
		
		whr = sum((1-self.model.ItemPrior[evidenceBasket]) * hits)/sum(1-self.model.ItemPrior[evidenceBasket])
		return whr

def help():
	print  'python2.6 -a [alpha] -t [teta] -nb [number of basket] -nr [number of recommendation] (-h print this help)'

if __name__ == '__main__':
	
#	try :
#		opts, args = getopt.getopt(sys.argv[1:], 'a:t:b:r:h', ['alpha=', 'teta=', 'nbBasket=', 'nbRecommendedItems=', 'help'])
#		print opts
#	except getopt.GetoptError:
#		help()
#		sys.exit(2)
#	
#	for opt, arg in opts:
#		if opt in ('-h', '--help'):
#			help()
#			sys.exit()
#		elif opt in ('-a', '--alpha'):
#			alpha = float(arg)
#		elif opt in ('-t', '--teta'):
#			teta = float(arg)
#		elif opt in ('-b', '--nbBasket'):
#			nbBasket = int(arg)
#		elif opt in ('-r', '--nbRecommendedItems'):
#			nbReco = int(arg)
#	
#	print 'Parameters: alpha %s, teta %s, nbBasket %s, nbReco %s'%(alpha, teta, nbBasket, nbReco)
	
	###############################################################
	# PARSE DATA
	###############################################################
	print 'Parse Data'
	path = path = '/Users/kfrancoi/These/Data/RetailData/TaFengDataSet/'
	dataPaths = [path+'D01', path+'D11', path+'D12']
	data = BasketPreprocessing()
	if os.path.exists(data.saveUI):
		print 'Train data exists --> loading files...'
		data.loadTrainData()
	else:
		print 'Train data does not exist --> parsing files...'
		data.trainDataPreprocessing(dataPaths)
		data.saveTrainData()
	###############################################################
	# CREATE MODELS
	###############################################################
#	print 'Create the model based on the training set'
#	
#	modelSOP = SOPRecoModel(data.getUserItemMatrix(), alpha, teta)
#	modelSOP.launch()
#	modelCosine = CosineRecoModel(data.getUserItemMatrix())
#	modelCondProb = CondProbRecoModel(data.getUserItemMatrix())
#	modelRW = BasketRandomWalk(data.getUserItemMatrix())
	
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
	
	###############################################################
	# SET RECOMMENDATION
	###############################################################
#	if nbBasket == -1:
#		evalSOP = Evaluation(modelSOP, data.getBasketItemList(), nbReco)
#		evalCosine = Evaluation(modelCosine, data.getBasketItemList(), nbReco)
#		evalCondProb = Evaluation(modelCondProb, data.getBasketItemList(), nbReco)
#		evalRW = Evaluation(modelRW, data.getBasketItemList(), nbReco)
#	else :
#		evalSOP = Evaluation(modelSOP, data.getBasketItemList()[:nbBasket], nbReco)
#		evalCosine = Evaluation(modelCosine, data.getBasketItemList()[:nbBasket], nbReco)
#		evalCondProb = Evaluation(modelCondProb, data.getBasketItemList()[:nbBasket], nbReco)
#		evalRW = Evaluation(modelRW, data.getBasketItemList()[:nbBasket], nbReco)
	
	###############################################################
	# LAUNCH RECOMMENDATION + SAVE RESULTS
	###############################################################	
#	t = time.time()
#	evalSOP.newEval()
#	SOPTime = time.time()-t
#	mmwrite('SOPPerf',evalSOP.perf) 
#
#	t = time.time()
#	evalCosine.newEval()
#	CosineTime = time.time()-t
#	mmwrite('CosinePerf', evalCosine.perf)
#	
#	t = time.time()
#	evalCondProb.newEval()
#	CondProbTime = time.time()-t
#	mmwrite('CondProbPerf', evalCondProb.perf)
#	
#	t = time.time()
#	evalRW.newEval()
#	RWTime = time.time()-t
#	mmwrite('RWPerf', evalRW.perf)
#	
#	
#	print 'SOP Execution time:', SOPTime
#	print 'Cosine Execution time:', CosineTime
#	print 'Conditional Probability Execution time:', CondProbTime
#	print 'Random Walk Execution time:', RWTime
