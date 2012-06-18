"""
Cython module for distance matrix computation
Author
"""

import numpy as np
cimport numpy as np

cdef extern from "math.h":
	double sqrt(double x)
	
cdef extern from "math.h":
	double log(double x)


def sqrtEuclidean(np.ndarray[np.float64_t, ndim=2] X):
	"""
	Compute the distance matrix from a similarity matrix X
	
	Parameters:
	----------
	@param X : A two dimensional numpy array
	@type X : np.array float
	
	Returns:
	--------
	An numpy array being the distance matrix
	"""
	cdef int i,j
	
	cdef int row = X.shape[0]
	cdef int col = X.shape[1]
	
	cdef np.ndarray[np.float64_t, ndim=2] C = np.zeros_like(X)
	
	for i in range(row):
		for j in range(col):
			C[i,j] = sqrt(X[i,i] + X[j,j] - 2*X[i,j])
	
	return C
	
def tf(np.ndarray[np.float64_t, ndim=2] X):
	"""
	Term frequency function. Assuming X is a matrix of User-Item
	
	Parameters:
	----------
	@param X : A two dimensional numpy array
	@type X : np.array float
	
	Returns:
	--------
	An numpy array being the tf normalized matrix
	"""
	
	cdef int u, i
	cdef int row = X.shape[0]
	cdef int col = X.shape[1]
	
	cdef np.ndarray[np.float64_t, ndim=2] tfX = np.zeros_like(X)
	
	for u in xrange(row):
		for i in xrange(col):
			tfX[u,i] = X[u,i] / nnz(X[u,:])
	
	return tfX
	
def nnz(np.ndarray[np.float64_t, ndim=1] X):
	"""
	Count the number of zeros in a vector of float
	
	Parameters:
	----------
	@param X : A one dimensional numpy array
	@type X : np.array float
	
	Returns:
	--------
	An scalar being the the number of non zero element
	"""
	
	cdef int n = 0
	cdef int i
	cdef int elem = X.shape[0]
	
	for i in xrange(elem):
		if X[i] != 0:
			n+=1
	
	return n
	
def idf(np.ndarray[np.float64_t, ndim=2] X):
	"""
	Inverse document frequency. Assuming X is a matrix of User-Item
	
	Parameters:
	----------
	@param X : A two dimensional numpy array
	@type X : np.array float
	
	Returns:
	--------
	An numpy 1d array being the idf normalized vector
	"""
	
	cdef int i
	cdef int elem = X.shape[1]
	cdef int totBasket = X.shape[0]
	
	cdef np.ndarray[np.float64_t, ndim=1] idfX = np.zeros((elem))
	for i in xrange(elem):
			idfX[i] = log( totBasket / nnz(X[:,i]))
	
	return idfX
