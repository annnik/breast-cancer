from __future__ import division
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import math

#read data into array
data = np.genfromtxt('cancer.txt',delimiter=',')
x=data[:,0]
y=data[:,1]
m=len(y)
a=91
#a=np.ones(m,2)


def cost(x, y, theta=np.zeros((2,1))):
	"""Computes the cost of linear regression
	theta = parameter for linear regression
	x and y are the data points
	This is done to monitor the cost of gradient descent"""
	m = len(x)
	J = 1/(2*m) * sum((x.dot(theta).flatten()- y)**2)
	return J


def gradientDesc(x, y, theta=np.zeros((2,1)), alpha=.01,iterations=1500):
	""""Gradient Descent implementation of 
	linear regression with one variable"""
	m = y.size
	J = []
	for numbers in range(iterations):
		a = theta[0][0] - alpha*(1/m)*sum((x.dot(theta).flatten() - y)*x[:,0])
		b = theta[1][0] - alpha*(1/m)*sum((x.dot(theta).flatten() - y)*x[:,1])
		theta[0][0],theta[1][0]=a,b
		print theta[0][0]
		print theta[1][0]
		J.append(cost(x,y,theta))
		print 'Cost: ' + str(J[-1])
	return theta

#scatterplot of data with option to save figure.
def scatterPlot(x,y,yp=None,savePng=False):
	plt.xlabel('Clump Thickness')
	plt.ylabel('Marginal Adhension')
	plt.scatter(x, y, marker='x')
	if yp != None:
		plt.plot(x,yp)
	if savePng==false:
		plt.show()
	else:
		name = raw_input('Name Figure File: ')
		plt.savefig(name+'.png')
		

