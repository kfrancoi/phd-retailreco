'''
Created on May 11, 2010

@author: kfrancoi
'''
import getopt
import sys
import os
import time
import matplotlib.pyplot as plt
import processing3 as processing #Modif!
from numpy import *
from scipy.io import mmwrite, mmread


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
		plt.savefig('../results/TestBasket_SOP %s alpha %s.pdf'%(label,a))

def testCosine(UI, evidence=[0,3], label=''):
	
	print 'Recommendation model'
	model = processing.CosineRecoModel(UI)
	b = model.getScores(evidence, -1)
	popInd = flipud(argsort(model.ItemPop))
	plt.bar(range(len(b[popInd])),b[popInd])
	bInd = argsort(b[:])
	plt.title('Cosine : %s'%bInd[-3:])
	plt.savefig('../results/TestBasket_Cosine %s.pdf'%(label))

def load():
	
	###############################################################
	# PARSE DATA
	###############################################################
	print 'Parse Data'
	path = path = '/Users/kfrancoi/These/Data/RetailData/TaFengDataSet/'
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


def SOP(alpha, teta, nbBasket, nbReco):
	
	data = load()
	###############################################################
	# CREATE MODELS
	###############################################################
	print 'Create the model based on the training set'
	
	modelSOP = processing.SOPRecoModel(data.getUserItemMatrix(), alpha, teta)
	modelSOP.launch()
		
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
	mmwrite('SOPPerf_a%s_t%s_nb%s_nr%s'%(alpha,teta,nbBasket,nbReco),evalSOP.perf) 
	
	print 'SOP Execution time:', SOPTime
	print 'Performances : '
	print evalSOP.testNames
	print evalSOP.meanPerf()
	evalSOP.savePerf('SOPPerf_a%s_t%s_nb%s_nr%s.txt'%(alpha,teta,nbBasket,nbReco))
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
	mmwrite('Cosine_nb%s'%nbBasket,evalCosine.perf) 
	
	print 'Cosine Execution time:', CosineTime
	print 'Performances :'
	print evalCosine.testNames
	print evalCosine.meanPerf()
	evalCosine.savePerf('Cosine_nb%s.txt'%nbBasket)
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
	mmwrite('CondProb_a%s_nb%s'%(alpha, nbBasket),evalCondProb.perf) 
	
	print 'Cond Prob Execution time:', CondProbTime
	print 'Performances :'
	print evalCondProb.testNames
	print evalCondProb.meanPerf()
	evalCondProb.savePerf('CondProb_a%s_nb%s.txt'%(alpha, nbBasket))
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
	mmwrite('RW_a%s_nb%s'%(alpha, nbBasket),evalRW.perf) 
	
	print 'RW Execution time:', RWTime
	print 'Performances :'
	print evalRW.testNames
	print evalRW.meanPerf()
	evalRW.savePerf('RW_a%s_nb%s.txt'%(alpha, nbBasket))
	return evalRW

def RWWR(alpha, nbBasket, nbReco):
	data = load()
	###############################################################
	# CREATE MODELS
	###############################################################
	print 'Create the model based on the training set'
	
	modelRWWR = processing.RandomWalkWithRestartRecoModel(data.getUserItemMatrix(), alpha)
		
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
	mmwrite('RWWR_a%s_nb%s'%(alpha, nbBasket),evalRWWR.perf) 
	
	print 'RWWR Execution time:', RWWRTime
	print 'Performances :'
	print evalRWWR.testNames
	print evalRWWR.meanPerf()
	evalRWWR.savePerf('RWWR_a%s_nb%s'%(alpha, nbBasket))
	return evalRWWR


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
	mmwrite('Pop',evalPop.perf) 
	
	print 'Pop Execution time:', PopTime
	print 'Performances :'
	print evalPop.testNames
	print evalPop.meanPerf()
	evalPop.savePerf('Pop.txt')
	return evalPop


def help():
	print 'python2.6 Experiences.py Cosine -b -r'
	print 'pythno2.6 Experiences.py CondProb -a -b -r'
	print 'python2.6 Experiences.py RW -a -b -r'
	print 'python2.6 Experiences.py SOP -a -t -b -r'
	print 'python2.6 Experiences.py Pop -b -r'
	print 'python2.6 Experiences.py RWWR -a -b -r'


if __name__ == '__main__':
	
	if sys.argv > 1:
		#######################################################################
		#SOP
		#######################################################################
		if sys.argv[1] == 'SOP':
			try :
				opts, args = getopt.getopt(sys.argv[2:], 'a:t:b:r:h', ['alpha=', 'teta=', 'nbBasket=', 'nbRecommendedItems=', 'help'])
				print opts
			except getopt.GetoptError:
					help()
					sys.exit(2)
	
			for opt, arg in opts:
				if opt in ('-h', '--help'):
					help()
					sys.exit()
				elif opt in ('-a', '--alpha'):
					alpha = float(arg)
				elif opt in ('-t', '--teta'):
					teta = float(arg)
				elif opt in ('-b', '--nbBasket'):
					nbBasket = int(arg)
				elif opt in ('-r', '--nbRecommendedItems'):
					nbReco = int(arg)
			
			SOP(alpha, teta, nbBasket, nbReco)
		
		#RW
		#######################################################################	
		elif sys.argv[1] == 'RW':
			try:
				opts, args = getopt.getopt(sys.argv[2:], 'a:b:r:', ['alpha=', 'teta=', 'nbBasket=', 'nbRecommendedItems=', 'help'])
				print opts
			except getopt.GetoptError:
				help()
				sys.exit(2)
			
			for opt, arg in opts:
				if opt in ('-h', '--help'):
					help()
					sys.exit()
				elif opt in ('-a', '--alpha'):
					alpha = float(arg)
				elif opt in ('-b', '--nbBasket'):
					nbBasket = int(arg)
				elif opt in ('-r', '--nbRecommendedItems'):
					nbReco = int(arg)
			
			RW(alpha, nbBasket, nbReco)
		
		#CondProb
		#######################################################################	
		elif sys.argv[1] == 'CondProb':
			try:	
				opts, args = getopt.getopt(sys.argv[2:], 'a:b:r:', ['alpha=', 'nbBasket=', 'nbRecommendedItems=', 'help'])
				print opts
			except getopt.GetoptError:
				help()
				sys.exit(2)
			
			for opt, arg in opts:
				if opt in ('-h', '--help'):
					help()
					sys.exit()
				elif opt in ('-a', '--alpha'):
					alpha = float(arg)
				elif opt in ('-b', '--nbBasket'):
					nbBasket = int(arg)
				elif opt in ('-r', '--nbRecommendedItems'):
					nbReco = int(arg)

			CondProb(alpha, nbBasket, nbReco)
		
				
		#Random Walk With Restart
		#######################################################################	
		elif sys.argv[1] == 'RWWR':
			try:	
				opts, args = getopt.getopt(sys.argv[2:], 'a:b:r:', ['alpha=', 'nbBasket=', 'nbRecommendedItems=', 'help'])
				print opts
			except getopt.GetoptError:
				help()
				sys.exit(2)
			
			for opt, arg in opts:
				if opt in ('-h', '--help'):
					help()
					sys.exit()
				elif opt in ('-a', '--alpha'):
					alpha = float(arg)
				elif opt in ('-b', '--nbBasket'):
					nbBasket = int(arg)
				elif opt in ('-r', '--nbRecommendedItems'):
					nbReco = int(arg)

			RWWR(alpha, nbBasket, nbReco)

		
		#Cosine
		#######################################################################
		elif sys.argv[1] == 'Cosine':
			try:	
				opts, args = getopt.getopt(sys.argv[2:], 'b:r:', ['nbBasket=', 'nbRecommendedItems=', 'help'])
				print opts
			except getopt.GetoptError:
				help()
				sys.exit(2)
			
			for opt, arg in opts:
				if opt in ('-h', '--help'):
					help()
					sys.exit()
				elif opt in ('-b', '--nbBasket'):
					nbBasket = int(arg)
				elif opt in ('-r', '--nbRecommendedItems'):
					nbReco = int(arg)
			
			Cosine(nbBasket, nbReco)
		
		elif sys.argv[1] == 'Pop':
			try:	
				opts, args = getopt.getopt(sys.argv[2:], 'b:r:', ['nbBasket=', 'nbRecommendedItems=', 'help'])
				print opts
			except getopt.GetoptError:
				help()
				sys.exit(2)
			
			for opt, arg in opts:
				if opt in ('-h', '--help'):
					help()
					sys.exit()
				elif opt in ('-b', '--nbBasket'):
					nbBasket = int(arg)
				elif opt in ('-r', '--nbRecommendedItems'):
					nbReco = int(arg)
			
			Pop(nbBasket, nbReco)
		

	
