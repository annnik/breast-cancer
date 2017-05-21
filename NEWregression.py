from random import seed
from random import randrange
from csv import reader
from math import sqrt
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
from pandas import date_range,Series,DataFrame,read_csv, qcut
savepred=list()

def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as f:
		csv_reader = reader(f)
		for row in csv_reader:
			if not row:
				continue
			else:
				dataset.append(row)
	return dataset

def str_column_to_float(dataset,column):
	for row in dataset:
		row[column] = float(row[column].strip())

def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax

def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0])/(minmax[i][1] - minmax[i][0])

def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split


def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		error = actual[i] - predicted[i]
		sq_error = error**2
		sum_error += sq_error
	mean_error = sum_error/float(len(actual))
	return (sqrt(mean_error))

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	#plt.plot(algorithm, 'ro')
	#plt.show()
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		rmse = rmse_metric(actual, predicted)
		scores.append(rmse)
	return scores

def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += row[i]*coefficients[i+1]

	return yhat

def coefficients_sgd(train, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			y = predict(row, coef)
			error = y - row[-1]
			coef[0] = coef[0] - l_rate*error
			for i in range(len(coef)-1):
				coef[i+1] = coef[i+1] - l_rate*error*row[i]
	return coef

def multivariate_linear_regression(train, test, l_rate, n_epoch):
	predictions = list()
	coef = coefficients_sgd(train, l_rate, n_epoch)
	for row in test:
		yhat = predict(row, coef)
		predictions.append(yhat)
	print('predictions: %s' % predictions)
	plt.plot(predictions,'o')
	plt.show()
	savepred=predictions
	return(predictions)


seed(1)
filename = 'brcancer.csv'
dataset = load_csv(filename)
# df=read_csv(filename, header = None, names = ['x','y'])
#
# x = np.array(df.x)
# y = np.array(df.y)
#
# theta = np.zeros((2,1))
# #scatterplot of data with option to save figure.
# def scatterPlot(x,y,yp=None,savePng=False):
# 	plt.xlabel('Clump Thickness')
# 	plt.ylabel('Marginal Adhension')
# 	plt.scatter(x, y, marker='x')
# 	#if yp is not None:
# 		#plt.plot(x,yp)
# 	#if savePng==False:
# 		#plt.show()
# 	#else:
# 		#name = raw_input('Name Figure File: ')
# 		#plt.savefig(name+'.png')
#
#
# #scatterPlot(x,y)
#
# #linear regression implementation using libraries
# (m,b) = np.polyfit(x,y,1)
# print 'Slope is ' + str(m)
# print 'Y intercept is ' + str(b)
# yp = np.polyval([m,b],x)
# scatterPlot(x,y,yp)
#


#plt.plot(dataset, 'ro')
#plt.show()

for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)



#plt.plot(dataset, 'ro')
#plt.show()

n_folds = 5
l_rate = 0.01
n_epoch = 50



scores = evaluate_algorithm(dataset, multivariate_linear_regression, n_folds, l_rate, n_epoch)

# names=['id','ClumpThickness','UniformityofCellSize','UniformityofCellShape',	'MarginalAdhesion',	'SingleEpithelialCellSize',	'BareNuclei',	'BlandChromatin',	'NormalNucleoli',	'Mitoses',	'Class']
# data=read_csv(filename, delimiter='\t',names=names).dropna()
# plt.scatter(data.Mitoses, data.Class)
# plt.xlabel("Number of Votes")
# plt.ylabel("IMDB Rating")
# plt.xscale('log')

#plt.plot(dataset,'o')
#plt.plot(predictions,'o')
#plt.show()

plt.plot(scores)
plt.show()

print('Scores: %s' % scores)
print('Mean RMSE: %.3f' % (sum(scores)/float(len(scores))))

