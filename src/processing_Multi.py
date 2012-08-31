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

import visualization

#from IPython.Debugger import Tracer
from numpy.core.numeric import zeros_like

#debug_here = Tracer()

sys.path.append("../libraries/plsa")
import plsa 

class TransactionPreprocessing:
	
	def __init__(self, label=''):
		self.userDict = dict()
		self.itemDict = dict()
		self.basketDict = {}
		self.saveUI = '%s_sUI.mtx'%label
		self.saveItemDict = '%s_itemDict.dict'%label
		self.saveBI = '%s_basketDict.dict'%label

	def readDataFile(self, dataPaths, delimiter=';', skiprows=1, usecols=(1,3), dtype=None):
		'''
		This method read one or multiple file and return a transaction dataset.
		dtype allows to better define the type and name of each column
		ex : 		
		
		event_fields = ['datetime', 'customerID', 'item']
		event_format = ['S20', 'int', 'int']
		dt=dtype({'names': [x for x in event_fields], 'formats': [b for b in event_format]})
		'''
		
		datas = list()
		for path in dataPaths:
				print 'Parsing data %s'%path
				if dtype == None:
					datas.append(loadtxt(path, delimiter=delimiter, skiprows=skiprows, usecols = usecols))
				else :
					datas.append(loadtxt(path, delimiter=delimiter, skiprows=skiprows, usecols = usecols, dtype=dtype))
		transData = datas[0]
		
		for data in datas[1:]:
			transData = concatenate((transData, data))
		
		return transData

	def aggregateUserProcessing(self, dataset):
		'''
		This method reads a transaction dataset and transforms it into a aggregate matrix
		Transaction dataset is made of:
		Datetime
		User_id
		Item_id
		'''

		if len(dataset[0]) != 3:
			raise Exception('The dataset must contains two columns')
		
		ind_u = 0
		ind_i = 0
		print('Counting user and item...')
		for d, u, i in dataset:
			if int(u) not in self.userDict.keys():
				self.userDict[int(u)] = ind_u
				ind_u+=1
			if i not in self.itemDict.keys():
				self.itemDict[i] = ind_i
				ind_i+=1
		
		self.UI = zeros((len(self.userDict.keys()), len(self.itemDict.keys())))
		
		print('Populating aggregate matrix')
		for d, u, i in dataset:
			self.UI[self.userDict[int(u)], self.itemDict[i]]+=1
		
		# Log frequence UI transformation (TO BE OPTIMIZE)
		# -------------------------------
		print 'Frequence normalisation'
		for i,j in argwhere(self.UI!=0):
			self.UI[i,j] = log(self.UI[i,j]+1)
		
		return self.UI
	
	def aggregateUserProcessing2C(self, dataset):
		'''
		This method reads a transaction dataset and transforms it into a aggregate matrix
		Transaction dataset is made of:
		User_id
		Item_id
		'''

		if len(dataset[0]) != 2:
			raise Exception('The dataset must contains two columns')
		
		ind_u = 0
		ind_i = 0
		print('Counting user and item...')
		for u, i in dataset:
			if int(u) not in self.userDict.keys():
				self.userDict[int(u)] = ind_u
				ind_u+=1
			if i not in self.itemDict.keys():
				self.itemDict[i] = ind_i
				ind_i+=1
		
		self.UI = zeros((len(self.userDict.keys()), len(self.itemDict.keys())))
		
		print('Populating aggregate matrix')
		for u, i in dataset:
			self.UI[self.userDict[int(u)], self.itemDict[i]]+=1
		
		# Log frequence UI transformation (TO BE OPTIMIZE)
		# -------------------------------
		#print 'Frequence normalisation'
		#for i,j in argwhere(UI!=0):
		#	UI[i,j] = log(UI[i,j]+1)
		
		return self.UI
	
	def aggregateTransactionProcessing(self, dataset):
		'''
		This method reads a transaction dataset and transforms it into a transaction aggregate matrix.
		dataset is a list of tuple of the form : date  user  item
		'''
		
		if len(dataset[0]) != 3:
			raise Exception('The dataset must contains 3 columns')
		
		print 'Create test baskets dictionnary'
		
		basketDict = dict()
		
		ind_b = 0
		for i in arange(len(dataset)):
			if (dataset[i][0],dataset[i][1]) not in basketDict.keys():
				basketDict[(dataset[i][0],dataset[i][1])] = ind_b
				ind_b+=1
		
		print 'Create basket array'
		BasketItem = zeros((len(basketDict), len(self.itemDict.keys())), dtype='int')
		for i in arange(len(dataset)):
			if dataset[i][2] in self.itemDict.keys():
				BasketItem[basketDict[(dataset[i][0], dataset[i][1])], self.itemDict[dataset[i][2]]] +=1
		
		
		BasketItemList = []
		for i in xrange(BasketItem.shape[0]):
			BasketItemList.append(argwhere(BasketItem[i,:]!=0).flatten())
		
		return BasketItemList
	
	def aggregateTransactionProcessing2C(self, dataset):
		'''
		This method reads a transaction dataset and transforms it into a transaction aggregate matrix.
		dataset is a list of tuple of the form : date  user  item
		dataset is made of:
		- User_id
		- Item_id
		'''
		
		if len(dataset[0]) != 2:
			raise Exception('The dataset must contains 2 columns')
		
		print 'Create test baskets dictionnary'
		
		self.basketDict = dict()
		
		ind_b = 0
		for i in arange(len(dataset)):
			if dataset[i][0] not in self.basketDict.keys():
				self.basketDict[dataset[i][0]] = ind_b
				ind_b+=1
		
		print 'Create basket array'
		BasketItem = zeros((len(self.basketDict), len(self.itemDict.keys())), dtype='int')
		for i in arange(len(dataset)):
			if dataset[i][1] in self.itemDict.keys():
				BasketItem[self.basketDict[dataset[i][0]], self.itemDict[dataset[i][1]]] +=1
		
		
		BasketItemList = []
		for i in xrange(BasketItem.shape[0]):
			BasketItemList.append(argwhere(BasketItem[i,:]!=0).flatten())
		
		return BasketItemList
	
	def saveItemIDDict(self, fname):
		print 'Writing Baskets to file'
		file = open(fname, 'w')
		pickle.dump(self.itemDict, file)
		file.close()
		
	def loadItemIDDict(self, fname):
		print 'Read BI from file'
		file = open(fname, 'r')
		self.itemDict = pickle.load(file)
		file.close()
	
	def saveCustomerIDDict(self, fname):
		print 'Writing Baskets to file'
		file = open(fname, 'w')
		pickle.dump(self.basketDict, file)
		file.close()
		
	def loadCustomerIDDict(self, fname):
		print 'Read BI from file'
		file = open(fname, 'r')
		self.basketDict = pickle.load(file)
		file.close()
	
	def saveBasketData(self, fname, BasketItemList):
		print 'Writing Baskets to file'
		file = open(fname, 'w')
		self.BasketItemList = BasketItemList
		pickle.dump(BasketItemList, file)
		file.close()		
		
	def loadBasketData(self, fname):
		print 'Read BI from file'
		file = open(fname, 'r')
		BasketItemList = pickle.load(file)
		file.close()
		self.BasketItemList = BasketItemList
		return BasketItemList
	
	def saveTrainData(self, fname, UI):
		print 'Writing UI to file'
		self.UI = UI
		sUI = sparse.csr_matrix(UI)
		mmwrite(fname, sUI)
	
	def loadTrainData(self, fname):
		print 'Read data from file'
		sUI = mmread(fname)
		self.UI = sUI.toarray()
		return sUI.toarray()
	
	def getUserItemMatrix(self):
		return self.UI
	
	def getBasketItemList(self):
		return self.BasketItemList
	
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
		self.UI = UI
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
		
		print 'Normalize User-Item matrix'
		self.nUI = matrix(UI) / sum(matrix(UI),1)

class BasketRandomWalk():
	'''
	This class build a basket sensitive random walk model on bipartite network. Supposedly the same
	technique as the Random Walk With Restart.
	'''
	def __init__(self, recoUI, d=0.8):
		self.UI = recoUI.UI
		self.d = d
		
		self.ItemPop = recoUI.ItemPop
		self.ItemPrior = recoUI.ItemPrior
		self.nbUsers = recoUI.nbUsers
		self.nbItems = recoUI.nbItems
		self.nUI = recoUI.nUI
		
		self._buildRandomWalkScore(d)
	
	def __repr__(self):
		return '<BaskerRandomWalk: alpha %s>'%self.d
	
	def _buildRandomWalkScore(self,d):
		
		epsilon = 0.00001
		#TO CHECK : Should P be computed based on the cosine similarity matrix ??? I don't think so!
		#P = matrix(toolBox.randomWalkTransitionProbability(self.UI))
		#P = matrix(toolBox.AtoP(toolBox.cosineSimilarity(self.UI)))
		P = matrix(toolBox.transitionProbability(self.UI))
		R = matrix(zeros((self.nbItems, self.nbItems)), dtype=float64)
		U = matrix(eye(self.nbItems, self.nbItems,dtype=float64))
		#U = U / self.nbItems
		diff = 1000
		#count = 20
		print 'Convergence ...'
		while (diff > epsilon) :#and (count > 0):
			sys.stdout.write('.')
			oldRSum= sum(R)
			R = d*P*R + (1-d)*U
			diff = abs(oldRSum - sum(R)) / sum(R)
			#count-=1
			print diff
		
		#for i in arange(R.shape[0]):
		#	R[i,i] = 1
		print R
		print sum(R,0)
		self.R = array(R)
	
	def recommend(self, evidences, n):
		self.scores = sum(self.R[:, evidences],1)
		self.scores[evidences] = 0
		
		ind = argsort(self.scores)
		
		if n == -1:
			return ind
		else:
			return ind[-n:] #Return the n largest scores
		
class BasketRandomWalk_POP():
	'''
	This class build a basket sensitive random walk model on bipartite network. Supposedly the same
	technique as the Random Walk With Restart.
	'''
	def __init__(self, recoUI, d=0.8):
		self.UI = recoUI.UI
		self.d = d
		
		self.ItemPop = recoUI.ItemPop
		self.ItemPrior = recoUI.ItemPrior
		self.nbUsers = recoUI.nbUsers
		self.nbItems = recoUI.nbItems
		self.nUI = recoUI.nUI
		
		self._buildRandomWalkScore(d)
	
	def __repr__(self):
		return '<BaskerRandomWalk: alpha %s>'%self.d
	
	def _buildRandomWalkScore(self,d):
		
		epsilon = 0.00001
		#TO CHECK : Should P be computed based on the cosine similarity matrix ??? I don't think so!
		#P = matrix(toolBox.randomWalkTransitionProbability(self.UI))
		#P = matrix(toolBox.AtoP(toolBox.cosineSimilarity(self.UI)))
		P = matrix(toolBox.transitionProbability(self.UI))
		Phi = self.ItemPrior
		W = repeat(array([Phi]), Phi.shape[0], axis=0)
		#TO BE TESTED
		#W = 1.0/W
		W = exp(-500*W)
		
		P = matrix(array(P)*W)		
		R = matrix(zeros((self.nbItems, self.nbItems)), dtype=float64)
		U = matrix(eye(self.nbItems, self.nbItems,dtype=float64))
		#U = U / self.nbItems
		diff = 1000
		#count = 20
		print 'Convergence ...'
		while (diff > epsilon) :#and (count > 0):
			sys.stdout.write('.')
			oldRSum= sum(R)
			R = d*P*R + (1-d)*U
			diff = abs(oldRSum - sum(R)) / sum(R)
			#count-=1
			print diff
		
		#for i in arange(R.shape[0]):
		#	R[i,i] = 1
		print R
		print sum(R,0)
		self.R = array(R)
	
	def recommend(self, evidences, n):
		self.scores = sum(self.R[:, evidences],1)
		self.scores[evidences] = 0
		
		ind = argsort(self.scores)
		
		if n == -1:
			return ind
		else:
			return ind[-n:] #Return the n largest scores

class CondProbRecoModel():
	'''
	This class build a conditional probability similarity model and contains all information
	to make recommendation
	'''
	def __init__(self, recoUI, alpha=0.5):
		self.UI = recoUI.UI
		self.alpha = alpha
		self.ItemPop = recoUI.ItemPop
		self.ItemPrior = recoUI.ItemPrior
		self.nbUsers = recoUI.nbUsers
		self.nbItems = recoUI.nbItems
		self.nUI = recoUI.nUI
		self._buildSimMatrix()
	
	def __repr__(self):
		return '<CondProbRecoModel: alpha %s>'%self.alpha
	
	def _buildSimMatrix(self):
		#TOFIX : Is this working ???
		self.Sim= array(toolBox.condProbSimilarity2(self.nUI, self.UI, self.alpha))
		#self.Sim = toolBox.condProbSimilarity(self.UI, self.alpha)
	
	def recommend(self, evidences, n):
		self.scores = sum(self.Sim[evidences, :],0).flatten()
		self.scores[evidences] = 0
		
		ind = argsort(self.scores)
		if n == -1:
			return ind
		else:
			return ind[-n:] #Return the n largest scores

class CosineRecoModel():
	'''
	This class build a cosine and contain all information to make recommendation.
	cf(cos)
	'''
	def __init__(self, recoUI):
		self.UI = recoUI.UI
		self.ItemPop = recoUI.ItemPop
		self.ItemPrior = recoUI.ItemPrior
		self.nbUsers = recoUI.nbUsers
		self.nbItems = recoUI.nbItems
		self.nUI = recoUI.nUI
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

class SOPRecoModel():
	'''
	This class contain all model information on trained data
	cf(sop)
	'''
	def __init__(self, recoUI, alpha=0.5, teta=0.9):
		'''
		Init method
		@param UI: User-Item matrix
		'''
		self.UI = recoUI.UI #Log freq ???
		self.alpha = alpha
		self.teta = teta
		self.ItemPop = recoUI.ItemPop
		self.ItemPrior = recoUI.ItemPrior
		self.nbUsers = recoUI.nbUsers
		self.nbItems = recoUI.nbItems
		self.nUI = recoUI.nUI
	
	def __repr__(self):
		return '<SOPRecoModel: alpha %s teta %s>'%(self.alpha,self.teta)
	
	def computeSim(self, type='Cosine'):
		
		print 'Compute similarity'
		if type == 'cos':
			self.Sim = toolBox.cosineSimilarity(self.UI)
	
	def computeCostMatrix(self):
		
		print 'Compute cost matrix'
		
		#C = (1 - self.Sim) #OK because Sim 
		#C = toolBox.sqrtEuclideanDistance(self.Sim)
		#C = C/std(C)
		
		Phi = self.ItemPrior
		#Phi = log(1+Phi)
		#Phi = Phi / std(Phi)
		#visualization.itemWeight(Phi, 1)
		mPhi = repeat(array([Phi]), Phi.shape[0], axis=0)
		mPhi = mPhi/mPhi.max()
		
		#TODO : Normalize matrices
		#C = toolBox.normalizeMatrix(C)
		#mPhi = toolBox.normalizeMatrix(mPhi)
		#Ctot = self.alpha * C + (1-self.alpha) * mPhi
		
		#return Ctot
		return mPhi
	
	def train(self):
		#Split this methods into two parts to be able to quickly output Weight matrix for different values of alpha and teta
		
		#Cosine computation
		#self.computeSim('cos')
		
		#Compute cost matrix
		self.Ctot = array(self.computeCostMatrix())
		
		#Compute afinity matrix
		self.P = toolBox.transitionProbability(self.UI)
		#debug_here()
		#self.P = array(toolBox.AtoP(self.Sim))

		#Compute weight matrix
		#self.W = exp(-self.teta*self.Ctot) * self.P #TODO : Normalize matrices
		#self.W = exp(-self.Ctot*1000) * self.P
	
	def recommend(self, evidences, n):
		
		#BUG : As we expand the number of node to create a convinient Finite State Machine
		#we have to remove those node from the recomended items
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
		
		#Removing dummy nodes
		#print 'Betweeness before sorting'
		#print b
		b = sort(b)[:-len(evidences)]
		#print 'Betweeness after sorting'
		#print b
		b[evidences] = 0
		bInd = argsort(b[:,0])
		#print 'Len(bInd) : %s'%len(bInd)
		
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
		
		#Temporary modification
		U = zeros((self.P.shape[0], self.P.shape[0]))
		for i in nodeList:
			U[:,i] = 0
		self.Pref = self.alpha * self.P + (1-self.alpha) * U
		self.W = exp(-self.teta*self.Ctot) * self.Pref
		
		
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

class SOPRecoModel_Marco():
	'''
	This class contain all model information on trained data
	cf(sop)
	'''
	def __init__(self, recoUI, alpha=0.5, teta=0.9):
		'''
		Init method
		@param UI: User-Item matrix
		'''
		self.UI = recoUI.UI #Log freq ???
		self.alpha = alpha
		self.teta = teta
		self.ItemPop = recoUI.ItemPop
		self.ItemPrior = recoUI.ItemPrior
		self.nbUsers = recoUI.nbUsers
		self.nbItems = recoUI.nbItems
		self.nUI = recoUI.nUI
	
	def __repr__(self):
		return '<SOPRecoModel: alpha %s teta %s>'%(self.alpha,self.teta)
	
	def computeSim(self, type='Cosine'):
		
		print 'Compute similarity'
		if type == 'cos':
			self.Sim = toolBox.cosineSimilarity(self.UI)
	
	def computeCostMatrix(self):
		
		print 'Compute cost matrix'
		
		#C = (1 - self.Sim) #OK because Sim 
		#C = toolBox.sqrtEuclideanDistance(self.Sim)
		#C = C/std(C)
		
		Phi = self.ItemPrior
		#Phi = log(1+Phi)
		#Phi = Phi / std(Phi)
		#visualization.itemWeight(Phi, 1)
		mPhi = repeat(array([Phi]), Phi.shape[0], axis=0)
		mPhi = mPhi/mPhi.max()
		
		#TODO : Normalize matrices
		#C = toolBox.normalizeMatrix(C)
		#mPhi = toolBox.normalizeMatrix(mPhi)
		#Ctot = self.alpha * C + (1-self.alpha) * mPhi
		
		#return Ctot
		return mPhi
	
	def train(self):
		#Split this methods into two parts to be able to quickly output Weight matrix for different values of alpha and teta
		
		#Cosine computation
		#self.computeSim('cos')
		
		#Compute cost matrix
		self.Ctot = array(self.computeCostMatrix())
		
		#Compute afinity matrix
		self.P = toolBox.transitionProbability(self.UI)
		#debug_here()
		#self.P = array(toolBox.AtoP(self.Sim))

		#Compute weight matrix
		#self.W = exp(-self.teta*self.Ctot) * self.P #TODO : Normalize matrices
		#self.W = exp(-self.Ctot*1000) * self.P
	
	def recommend(self, evidences, n):
		
		#BUG : As we expand the number of node to create a convinient Finite State Machine
		#we have to remove those node from the recomended items
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
		
		#Removing dummy nodes
		#print 'Betweeness before sorting'
		#print b
		b = sort(b)[:-len(evidences)]
		#print 'Betweeness after sorting'
		#print b
		b[evidences] = 0
		bInd = argsort(b[:,0])
		#print 'Len(bInd) : %s'%len(bInd)
		
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
		
		#Temporary modification
		U = zeros((self.P.shape[0], self.P.shape[0]))
		for i in nodeList:
			U[:,i] = 0
		self.Pref = self.alpha * self.P + (1-self.alpha) * U
		self.W = exp(-self.teta*self.Ctot) * self.Pref
		
		
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

class RandomWalkWithRestartRecoModel():
	'''
	transition probability is :
		sim == cos : cos normalized
		sim == cp : cp normalized
		sim == bn : first order P
	'''
	
	def __init__(self, recoUI, alpha, sim=None):
		self.UI = recoUI.UI
		self.alpha = alpha
		self.sim = sim
		self.ItemPop = recoUI.ItemPop
		self.ItemPrior = recoUI.ItemPrior
		self.nbUsers = recoUI.nbUsers
		self.nbItems = recoUI.nbItems
		self.nUI = recoUI.nUI
	
	def __repr__(self):
		return '<RandomWalkWithRestartRecoModel: alpha %s>'%self.alpha
	
	def train(self):
		'''Different ways of computing the transition probability matrix can be considered'''
		#TO FIX : Should P be computed from the similarity matrix or directly as transition probability from the graph ? 
		
		if (self.sim == 'cos'):
			self.Sim = toolBox.cosineSimilarity(self.UI)
			self.P = toolBox.AtoP(self.Sim)
		if (self.sim == 'cp'):
			self.Sim = toolBox.cosineSimilarity(self.UI)
			self.P = toolBox.AtoP(self.Sim)
		if (self.sim == 'bn'):
			self.P = toolBox.transitionProbability(self.UI)
		#print self.P
	
	def recommend(self, evidences, n):
		
		I = eye(self.nbItems)
		u = zeros(self.nbItems)
		u[evidences] = 1 #1/nbEvidences ???
		try:
			#(I-dP) R = (1-d) u or U
			r = spLinalg.cgs(I-(self.alpha * self.P),(1-self.alpha)*u)
		except linalg.LinAlgError:
			print 'Linear Algebra Error'
			raise linalg.LinAlgError
		
		r[0][evidences] = 0
		ind = argsort(r[0]) #Item with highest scores are recommended
		
		return ind[-n:]

class BiasedRandomWalkWithRestartRecoModel():
	'''
	Biaised Random Walk With Restart
	'''
	
	def __init__(self, recoUI, theta, alpha, sim='bn'):
		self.UI = recoUI.UI
		self.theta = theta
		self.alpha = alpha
		self.sim = sim
		self.ItemPop = recoUI.ItemPop
		self.ItemPrior = recoUI.ItemPrior
		self.nbUsers = recoUI.nbUsers
		self.nbItems = recoUI.nbItems
		self.nUI = recoUI.nUI
	
	def __repr__(self):
		return '<BRWWR: theta %s>'%self.theta
	
	def train(self):
		print "Start training..."
		
		#self.Cost = repeat(reshape(self.ItemPrior, (self.ItemPrior.shape[0],1)), self.ItemPrior.shape[0], axis=0)
		self.Cost = repeat(reshape(self.ItemPrior, (1,self.ItemPrior.shape[0])), self.ItemPrior.shape[0], axis=0)
		print self.Cost.shape
		
		if (self.sim == 'cos'):
			self.Pref = toolBox.cosineSimilarity(self.UI)
		if (self.sim == 'bn'):
			self.Pref = toolBox.transitionProbability(self.UI)
		print self.Pref.shape
		self.Pref = toolBox.AtoP(self.Sim)
		#self.P = toolBox.BRWWR_Comp(self.Pref, ones(self.Pref.shape), self.theta)
		#self.P = toolBox.BRWWR_Comp(self.Pref, self.Cost, self.theta)
		self.P = self.Pref
		#print "Training done!"
		#print self.P
		#Call BRWWR
		
#		#Method RW
#		diff = 1000
#		epsilon = 0.00001
#		d = self.alpha
#		R = matrix(zeros((self.nbItems, self.nbItems)), dtype=float64)
#		U = matrix(eye(self.nbItems, self.nbItems,dtype=float64))
#		#count = 20
#		print 'Convergence ...'
#		while (diff > epsilon) :#and (count > 0):
#			sys.stdout.write('.')
#			oldRSum= sum(R)
#			R = d*self.P*R + (1-d)*U
#			diff = abs(oldRSum - sum(R)) / sum(R)
#			#count-=1
#			print diff
#		
#		#for i in arange(R.shape[0]):
#		#	R[i,i] = 1
#		print R
#		print sum(R,0)
#		self.R = array(R)

#	def recommend(self, evidences, n):
#		self.scores = sum(self.R[:, evidences],1)
#		self.scores[evidences] = 0
#		
#		ind = argsort(self.scores)
#		
#		if n == -1:
#			return ind
#		else:
#			return ind[-n:] #Return the n largest scores
			
	def recommend(self, evidences, n):
		
		I = eye(self.nbItems)
		u = zeros(self.nbItems)
		u[evidences] = 1 #1/nbEvidences ???
		try:
			#(I-dP) R = (1-d) u or U
			r = spLinalg.cgs(I-(self.alpha * self.P),(1-self.alpha)*u)
		except linalg.LinAlgError:
			print 'Linear Algebra Error'
			raise linalg.LinAlgError
		
		r[0][evidences] = 0
		ind = argsort(r[0]) #Item with highest scores are recommended
		
		return ind[-n:]

class PLSARecoModel():
	
	def __init__(self, recoUI, k):
		self.UI = recoUI.UI
		self.k = k
		self.maxiter = 500
		self.eps=0.01
		self.ItemPop = recoUI.ItemPop
		self.ItemPrior = recoUI.ItemPrior
		self.nbUsers = recoUI.nbUsers
		self.nbItems = recoUI.nbItems
		self.nUI = recoUI.nUI
	
	def __repr__(self):
		return '<PLSA (Z:%s)>'%(self.k)
	
	def train(self):
		self.pLSAmodel = plsa.pLSA()
		self.pLSAmodel.train(self.UI.transpose(), self.k, self.maxiter, self.eps)
	
	def recommend(self, evidences, n):
		
		#Array of item bought
		evidencesVec = zeros_like(self.ItemPrior)
		evidencesVec[evidences] = 1
		
		p_z_newUser = self.pLSAmodel.folding_in(evidencesVec)
		
		p_item_z = self.pLSAmodel.p_w_z
		
		p_item_newUser = zeros((p_item_z.shape[0]))
		for i in range(p_item_z.shape[1]):
			p_item_newUser+=p_item_z[:,i] * p_z_newUser[0]
		
		
		ind = argsort(p_item_newUser)
		
		if n == -1:
			return ind
		else:
			return ind[-n:]		

class PopRecoModel():
	
	def __init__(self,recoUI):
		self.UI = recoUI.UI
		self.ItemPop = recoUI.ItemPop
		self.ItemPrior = recoUI.ItemPrior
		self.nbUsers = recoUI.nbUsers
		self.nbItems = recoUI.nbItems
		self.nUI = recoUI.nUI
	
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
	def __init__(self, model=None, testBasketsList=None, n=None):
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
		#self.nbBaskets = len(self.evidenceBaskets)
		self.n = n
		self.removedList=[] # List of basket with less than 4 items
		
		#Create list of existing test
		self.testNames= []			
		for member in dir(self):
			if member[0:6] == '_perf_': #A test method must begin with _perf_
				self.testNames.append(member)
	
	def getRecommendation(self, evidenceBasket, type):
		'''
		Returns a list of recommended item for the current basket list
		@type evidenceBasket : list()
		@param evidencebasket : a list of item bought by a customer
		
		@type type : string [novel, all]
		@param type: 	novel means that recommended item will have a lower popularity that the lowest popular item in the basket
						all means that recommended item doesn't take into account the novelty
		
		@rtype : list
		'''
		if type == 'all':
			recoItems = self.model.recommend(evidenceBasket, self.n)
		if type == 'novel':
			minPop = min(self.model.ItemPop[evidenceBasket])
			recoItems = self.model.recommend(evidenceBasket, -1)
			recoItems = [i for i in recoItems if self.model.ItemPop[i]<minPop][-self.n:]
		return recoItems
	
	def recommendAll(self, type):
		'''
		Get recommendation for all customer in the test dataset
		@type type : string [novel, all]
		@param type: 	novel means that recommended item will have a lower popularity that the lowest popular item in the basket
						all means that recommended item doesn't take into account the novelty
		'''
		
		recoItems = list()
		for basket in self.evidenceBaskets:
			if len(basket) > (self.n-1) :
				recoItems.append(self.getRecommendation(basket, type))
			else:
				recoItems.append([])
		return recoItems
			
	
	def newEval(self):
		
		print('Start Evaluating')
		count = 0
		ind = argsort(self.model.ItemPrior)
		self.meanMostPop = mean(self.model.ItemPrior[ind][-10:]) #10 most popular
		self.nbBaskets = len(self.evidenceBaskets)
		self.perf = zeros((self.nbBaskets, len(self.testNames)))
		for basket in arange(len(self.evidenceBaskets)):
			count+=1
			if count%1000 == 0 :
				print 'Basket number %s'%count
				print 'Performances so far: '
				print self.testNames
				print self.computePerf()
			if len(self.evidenceBaskets[basket]) >=4 :
				for test in arange(len(self.testNames)):
					self.perf[basket,test] = getattr(self, self.testNames[test])(self.evidenceBaskets[basket])
			else:
				self.removedList.append(basket)
		self.computePerf()
		
	def computePerf(self):
		
		self.meanPerf = zeros(self.perf.shape[1], dtype='float')
		for i in range(self.meanPerf.shape[0]):
			self.meanPerf[i] = mean(self.perf[:,i])
		return self.meanPerf
	
	def savePerf(self, fname):
		
		file = open(fname, 'w')
		file.write('%s\n'%self.model)
		file.write('%s\n'%self.testNames)
		file.write('%s\n'%self.meanPerf)
		file.close() 
	
	def writeRecomendation(self, recoList, basketDict, itemDict, fname):
		'''
		Write in a file the recommendation results from recoList
		@type recoList : list(list())
		@param recoList: list of recommended product for each cutomer basket.
		'''
		
		indexCustomerDict = dict((v,k) for k, v in basketDict.iteritems())
		indexItemDict = dict((v,k) for k,v in itemDict.iteritems())
		
		file = open(fname, 'w')
		for ind, reco in enumerate(recoList):
			for item in reco:
				file.write('%s,%s\n'%(indexCustomerDict[ind],indexItemDict[item]))
		file.close()
		
	def _perf_bHR_pop(self, evidenceBasket):
		'''
		binary hit rate using x least popular item. Recommended item are always less popular than
		the least popular item in the evidence basket.
		@param evidenceBasket: ItemPop test item representing a transaction 
		'''
		#print 'EvidenceBasket : %s'%evidenceBasket
		popInd = argsort(self.model.ItemPop[evidenceBasket])
		sortedEvidenceBasket = evidenceBasket[popInd] # TO FIX !!!!
		targets = sortedEvidenceBasket[:self.n] #Define targets according to pop
		evidences = [item for item in evidenceBasket if item not in targets] #Define evidences
		minPop = min(self.model.ItemPop[evidences])
		recoItems = self.model.recommend(evidences, -1)
		#print 'Max recoItems : %s'%max(recoItems)
		#print 'Min Pop : %s'%minPop
		#print 'len(Item Pop) : %s'%len(self.model.ItemPop)
		
		recoItems = [i for i in recoItems if self.model.ItemPop[i]<minPop][-self.n:] #recommend only product less popular than the less popular in the basket

		#print 'Evidences : %s'%evidences
		#print 'RecoItem : %s'%recoItems
		#print 'Targets : %s'%targets
		
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

		#print 'RecoItem : %s'%recoItems
		#print 'Evidences : %s'%evidences
		#print 'Targets : %s'%targets
		
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
	
	#def _perf_Recall(self, evidenceBasket):
	#	'''
	#	Recall is the fraction of relevant item that are retrieved
	#	Idem brr(pop, rnd) but account for the true proportion instead of binary rates.
	#	'''
	
	#def _perf_Precision(self, evidenceBasket):
	#	'''
	#	Precision is the fraction of retrieved items that are relevant
	#	'''
#	def _perf_Popularity(self, evidenceBasket):
#		'''
#		Popularity rate is the mean popularity of recommended items.
#		'''
#		popInd = argsort(self.model.ItemPop[evidenceBasket])
#		sortedEvidenceBasket = evidenceBasket[popInd] # TO FIX !!!!
#		targets = sortedEvidenceBasket[:self.n] #Define targets according to pop
#		evidences = [item for item in evidenceBasket if item not in targets] #Define evidences
#		minPop = min(self.model.ItemPop[evidences])
#		recoItems = self.model.recommend(evidences, -1)
#		recoItems = [i for i in recoItems if self.model.ItemPop[i]<minPop][-self.n:]
#		
#		return mean(self.model.ItemPrior[recoItems])
	
	def _perf_Novelty(self, evidenceBasket):
		'''
		Novelty rate is the difference in terms of novelty compare to the highest popular items.
		'''
		popInd = argsort(self.model.ItemPop[evidenceBasket])
		sortedEvidenceBasket = evidenceBasket[popInd]
		targets = sortedEvidenceBasket[:self.n]
		evidences = [item for item in evidenceBasket if item not in targets]
		minPop = min(self.model.ItemPop[evidences])
		recoItems = self.model.recommend(evidences, -1)		
		recoItems = [i for i in recoItems if self.model.ItemPop[i]<minPop][-10:]
		
		recoPop = self.model.ItemPrior[recoItems]
		#print 'Novelty rate : %s'%mean((self.meanMostPop - recoPop) / self.meanMostPop)
		return mean((self.meanMostPop - recoPop) / self.meanMostPop)
		

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
