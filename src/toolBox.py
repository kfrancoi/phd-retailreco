'''
Created on May 11, 2010

@author: kfrancoi
'''
import os
import distance
import scipy.sparse as sparse
from numpy import *


def cosineSimilarity(X):
	''' 
	Compute a cosine similarity matrix on a User-Item matrix X
	weight of matrix X is either a frequence or a rating
	'''
	sX = sparse.csr_matrix(X)
	K = sX.T * sX
	K = matrix(K.toarray())
	D = matrix(diag(power(diag(K),-0.5)))
	Sim = D * K * D
	
	#for i in arange(Sim.shape[0]):
	#	Sim[i,i] = 0
	
	return Sim

def condProbSimilarity(X, alpha):
	'''
	Compute a conditional probability matrix on a User-Item matrix X
	'''
	import copy as cp
	Xstd = cp.copy(X)
	Xstd[where(Xstd!=0)] = 1
	
	sXr = sparse.csr_matrix(Xstd)
	freqIJ = sXr.T * sXr
	
	freqI = sum(Xstd,0)
	freqJ = pow(freqI,alpha)
	freqI = matrix(freqI)
	freqIfreqJ = freqI.T * freqJ
	
	Sim = freqIJ.toarray() / array(freqIfreqJ)
	
	#for i in arange(Sim.shape[0]):
	#	Sim[i,i] = 0
	
	return Sim

def condProbSimilarity2(Xn, X, alpha):
		
	Rn = sum(Xn,0) #TO CHECK !!!
	Rn = tile(Rn, (X.shape[1],1))
	
	freqI = sum(X,0)
	freqJ = pow(freqI,alpha)
	freqI = freqI
	freqIfreqJ = freqI.T * freqJ
	
	Sim = Rn / array(freqIfreqJ)
	return Sim

def sqrtEuclideanDistance(X):
	'''
	Compute the square root euclidean distance matrix from a similarity matrix.
	This methods is implemented in Cython.
	'''
	return distance.sqrtEuclidean(X)

def sqrtEuclidieanDistanceSlow(X):
	'''
	Compute the sqrt distance matrix (or cost matrix) from a similarity matrix X
	'''
	row, col = X.shape
	C = zeros((row, col))
	for i in arange(row):
		for j in arange(col):
			C[i,j] = sqrt(X[i,i] + X[j,j] - 2*X[i,j])
	return C

def transitionProbability(X):
	'''
	Compute the transition probability matrix with a random walk perspective on the U-I bipartite network
	VALIDATED!
	'''
	row, col = X.shape
	colSum = sum(X,1)
	rowSum = sum(X,0)
	Xc = zeros(X.shape, dtype=float)
	Xr = zeros(X.shape, dtype=float)
	for i in arange(col):
		Xc[:,i] = X[:,i] / colSum
	for i in arange(row):
		Xr[i,:] = X[i,:] / rowSum
	
	sXc = sparse.csr_matrix(Xc)
	sXr= sparse.csr_matrix(Xr)
	TP = sXc.T * sXr
	
	for i in arange(TP.shape[0]):
		TP[i,i] = 0
	
	return TP.toarray()



#def firstOrdertransitionProb(X):
#	
#	P = zeros((X.shape[1], X.shape[1]))
#	for i in range(X.shape[1]):
#		for j in range(X.shape[1]):
#			for k in range(X.shape[0]):
#				P[j,i] += ((X[k,j]/sum(X[k,:]))*(X[k,i]/sum(X[:,i])))
#	
#	return P 

def AtoP(A):
	'''
	Transform an affinity matrix into a transition probability matrix
	'''
	A = array(A)
	s = sum(array(A),1)
	n = A.shape[0]
	e = ones((1,n))
	P = A*1.0 / array(matrix(s).T*matrix(e))
	#P = A*1.0 / s
	#for i in arange(P.shape[0]):
	#	P[i,i] = 0
	return P

def standardizeColumns(X) :
	'''returns (X - mean(X)) / std(X) '''

	m = mean(X)
	stdX = std(X)
	n = shape(X)[0]
	return (X - resize(m, (n,len(m))))/resize(stdX, (n,len(stdX)))


def normalizeMatrix(X):
	'''Return X/mean(X)'''
	
	return X/mean(X)


def tf(X):
	'''
	Term frequency function. Assuming X is a matrix of User-Item
	'''
	try:
		import distance2
		
		print("Cython tf...")
		return distance.tf(X)
	except :
		print("Careful !! Very slow tf python computation")
		tf = zeros_like(X)
		sX = sparse.csr_matrix(X)
		ind = sX.nonzero()
		for i in xrange(len(ind[0])):
			tf[ind[0][i],ind[1][i]] = sX[ind[0][i],ind[1][i]] / sX[ind[0][i],:].nnz
		
		return tf

def idf(X):
	'''
	Inverse document frequency. Assuming X is a matrix of User-Item
	'''
	try :
		import distance
		
		print('Cython idf...')
		return distance.idf(X)
	except :
		print("Careful !! Very slow idf python computation")
		idf = zeros((X.shape[1]))
		for i in xrange(idf.shape[0]):
				idf[i] = log( X.shape[0] / X[where(X[:,i]!=0),i].shape[1])
		return idf

def BRWWR_Comp(Pref, C, theta):
	'''Computation of the Biaised transition probability matrix'''
	
	P = zeros(Pref.shape)
	
	W = array(exp(-theta * matrix(C)))*Pref
	
	#return W
	[Dr, Vr] = linalg.eig(W)
	
	for i in xrange(Pref.shape[0]):
		for j in xrange(Pref.shape[1]):
			P[i,j] =  (Vr[j,0] * W[i,j]) / sum( Vr[:,0] * transpose(W[i,:]) )
	
	return P
	