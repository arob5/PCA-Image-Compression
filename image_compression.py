#
# image_compression.py
# Purpose: Perform dimensionality reduction on Olivetti face dataset
#
# Last Modified: 2/22/2018
# Modified By: Andrew Roberts
#

import run
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

def get_eigenstuff(X):
	""" Given a centered data matrix, returns the eigenvectors/values of the 
		covariance matrix 

	: Params
		X - nxd numpy array; each row represents an image
	: Returns
		w - 1-D numpy array of eigenvalues 
		v - dxd numpy arrar of eigenvectors
	""" 
	cov_mat = (X.T @ X) / X.shape[0]
	w, v = np.linalg.eigh(cov_mat)

	return w, v

def plot_eigenvalues(X, k):
	""" Plots a "scree plot" of the first k eigenvalues of X
		(aka eigenvalue decay of largest k eigenvalues)

	: Params
		X - nxd numpy array; each row represents an image
		k - Number of eigenvalues to plot
	"""
	w, v = get_eigenstuff(X)
	w = (w[-k:])[::-1]
	plt.scatter(np.arange(w.shape[0]), w)
	plt.xlabel("Component Number")
	plt.ylabel("Eigenvalue")
	plt.title("Eigenvalues of First {} Principal Components".format(k))
	plt.savefig("Eigenvalue_Plot.png")
	plt.show()

def plot_relative_error(X, l, u):
	""" Plots relative error for different low-rank approximations of X
		Relative error calculated using Frobenius Norm

	: Params
		X - nxd numpy array; each row represents an image
		l - Lowest rank relative error is calculated for
		u - Highest rank relative error is calculated for
	"""

	w, v = get_eigenstuff(X)
	
	rel_err = []
	for k in range(l, u):
		X_approx = get_dim_reduced_data(X, k, v) 
		rel_err.append(run.compute_relative_err(X, X_approx))

	plt.scatter(np.arange(u-l), rel_err)
	plt.xlabel("# Principal Components")
	plt.ylabel("Relative Error")
	plt.title("Relative Error of Approximation as Function of Number of Components")
	plt.savefig("Relative_Error_Plot.png")
	plt.show()
	
def visualize_eigenvectors(X):	
	""" Visualizes 15 largest eigenvectors reshaped as images

	: Params
		X - nxd numpy array; each row represents an image
	"""

	w, v = get_eigenstuff(X)
	v = v[:, -15:]

	nrows = 3
	ncols = 5
	fig, ax = plt.subplots(nrows, ncols)
	
	for k in range(nrows * ncols):
		i = k // ncols  # row index
		j = k % ncols  # col index
		ax[i][j].imshow(v[:,k].reshape(64, 64), cmap='gray')

	plt.show(block=False)
	
def get_dim_reduced_data(X, k, v=None):
	""" Return reduced rank approximation of the original dataset

	: Params
		X - nxd numpy array; each row represents an image
		k - Rank of low-rank approximation
		v - Eigenvectors (if they have already been calculated)

	: Returns
		X_reduced - Reduced rank approximation of X
	"""

	if v == None: 
		w, v = get_eigenstuff(X)

	# Method 1
	coefs = X @ v
	X_reduced = (coefs[:, -k:]) @ (v[:, -k:]).T 

	'''
	THESE METHODS ARE ALTERNATIVES TO METHOD 1 (GIVE SAME RESULT) 

	# Method 2: Projection Matrix
	v = v[:, -k:]
	proj_mat = v @ v.T
	X_reduced_2 = X @ proj_mat

	# Method 3: SVD
	U, S, Vt = np.linalg.svd(X, full_matrices=False) # S in DESCENDING ORDER
	X_reduced_3 = (U[:, :k] @ np.diag(S[:k])) @ Vt[:k, :]
	'''

	return X_reduced
		
if __name__ == "__main__":
	mu, X = run.run() 
	X_reduced = get_dim_reduced_data(X, 3)
	run.show_image_subset_reshaped(X_reduced)


