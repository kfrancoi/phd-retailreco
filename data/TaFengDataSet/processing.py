#!/usr/bin/env python
# encoding: utf-8
"""
untitled.py

Created by Kevin Françoisse on 2010-03-31.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.
"""

import sys
import os
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from numpy import *

def main():
	D01 = loadtxt('D01', delimiter=';', skiprows=1, usecols = (1,4))
	D02 = loadtxt('D02', delimiter=';', skiprows=1, usecols = (1,4))
	D11 = loadtxt('D11', delimiter=';', skiprows=1, usecols = (1,4))
	D12 = loadtxt('D12', delimiter=';', skiprows=1, usecols = (1,4))
	
	userDict = {}
	itemDict = {}

	# Hashing user and item ids
	# -------------------------
	ind_u = 0
	ind_i = 0
	for data in [D01]:#, D02, D11, D12]:
		print 'Parsing data...'
		for u,i in data:
			if int(u) not in userDict.keys():
				userDict[int(u)] = ind_u
				ind_u+=1
			if i not in itemDict.keys():
				itemDict[i] = ind_i
				ind_i+=1
	
	UI = zeros((len(userDict.keys()), len(itemDict.keys())))
	
	# Population of the transaction matrix UI
	# ---------------------------------------
	for data in [D01]:#, D02, D11, D12]:
		print 'Recording transactions...'
		for u,i in data:
			UI[userDict[int(u)], itemDict[i]]+=1
	
	# Sparse cosine based similarity
	# ------------------------------
	sUI = sparse.csr_matrix(UI, dtype='int32')
	K = sUI.T * sUI
	K = matrix(K.toarray())
	D = matrix(diag(power(diag(K),-0.5)))
	Sim = D * K * D
		
	plt.imshow(Sim)
	plt.show()
	

if __name__ == '__main__':
	main()

