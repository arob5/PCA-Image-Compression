import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces


def compute_relative_err(X, Xapprox):
		"""
		Computes the average entry-wise relative error between X and Xapprox.
		"""

		return np.linalg.norm(X - Xapprox) / np.linalg.norm(X)


def center_data(X):
		mu = X.mean(axis=0)
		X = X - mu
		return mu, X


def show_image_subset_reshaped(X):
		"""
		Visualize a subset of the images that are stored as rows of X.

		:param X: A N-by-d^2 matrix of N different d-by-d images, shaped as rows.
		"""
		nrows = 3
		ncols = 5
		fig, ax = plt.subplots(nrows, ncols)
		np.random.seed(0)  # for reproducing results
		random_indices = np.random.permutation(X.shape[0])[:nrows * ncols]
		for k in range(nrows * ncols):
				i = k // ncols  # row index
				j = k % ncols  # col index
				ax[i][j].imshow(X[random_indices[k]].reshape(64, 64), cmap='gray')
		plt.show(block=False)

def run(): 
		"""
		For purposes of calling from other files/Jupyter Notebook
		"""
		data = fetch_olivetti_faces()
		X = data.data
		mu, Xc = center_data(X)

		return mu, Xc

		

if __name__ == "__main__":

		# Below, the variable data is an object, and the attribute data.data is a
		# 400-by-4096 numpy array. Think of data.data as the N-by-d matrix of
		# datapoints from class, where d = 4096 = 64^2, is the number of pixels per
		# image. See documentation for fetch_olivetti_faces for more information.
		data = fetch_olivetti_faces()

		# Visualize a random selection of the face data
		X = data.data
		mu, Xc = center_data(X)
		show_image_subset_reshaped(X)
