import numpy as np
import pvml
import matplotlib.pyplot as plt

def minmax_normalization(Xtrain , Xtest): 
	xmin = Xtrain.min(0)
	xmax = Xtrain.max(0)
	Xtrain = (Xtrain - xmin) / (xmax - xmin) 
	Xtest = (Xtest - xmin) / (xmax - xmin) 
	return Xtrain , Xtest

def meanvar_normalization(Xtrain, Xtest):
	mu = Xtrain.mean(0)
	sigma = Xtrain.std(0)
	Xtrain = (Xtrain - mu) / sigma 
	Xtest = (Xtest - mu) / sigma 
	return Xtrain , Xtest

def maxabs_normalization(Xtrain , Xtest): 
	amax = np.abs(Xtrain).max(0)
	Xtrain = Xtrain / amax
	Xtest = Xtest / amax
	return Xtrain , Xtest

def whitening(Xtrain , Xtest):
	mu = Xtrain.mean(0)
	sigma = np.cov(Xtrain.T)
	evals , evecs = np.linalg.eigh(sigma) 
	w = evecs / np.sqrt(evals)
	Xtrain = (Xtrain - mu) @ w 
	Xtest = (Xtest - mu) @ w 
	return Xtrain , Xtest

def l2_normalization(X):
	q = np.sqrt((X ** 2).sum(1, keepdims=True))
	q = np.maximum(q, 1e-15) # 1e-15 avoids division by zero 
	X=X/q
	return X

def l1_normalization(X):
    q = np.abs(X).sum(1, keepdims=True)
    q = np.maximum(q, 1e-15) # 1e-15 avoids division by zero 	
    X=X/q
    return X

def l0_normalization(X):
    q = np.abs(X).max(1, keepdims=True)
    q = np.maximum(q, 1e-15) # 1e-15 avoids division by zero 	
    X=X/q
    return X
