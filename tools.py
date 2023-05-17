import os,csv,sys
import numpy as np
import pandas as pd
import scipy.stats
def mean_confidence_interval(data, confidence=0.9):

	data=np.array(data)

	a = 1.0 * data
	n = len(a)
	m, se = np.mean(a,axis=0), scipy.stats.sem(a,axis=0)
	# h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
	# h = scipy.stats.t.ppf(confidence, n - 1, loc = m, scale = se)
	h = scipy.stats.t.interval(confidence, n - 1, loc = m, scale = se)
	lower=h[0]
	upper=h[1]
	return m, lower, upper,h

def linearplot(xscale,x,lower,upper):
	"""Create a pyplot plot and save to buffer."""
	import matplotlib.pyplot as plt
	figure = plt.figure(figsize=(10,5))


	plt.plot(xscale,x,color='b')
	plt.plot(xscale,lower,color='r',linestyle='dotted')
	plt.plot(xscale,upper,color='r',linestyle='dotted')
	
	csfont = {'size'   : 14}

	plt.xlabel('Rc [m]')
	plt.ylabel('q [m$^3$/s/m]', **csfont)
	plt.yscale("log")
	plt.ylim((1E-8,1E-1))
	# plt.xlim((3,7.5))

	plt.grid(visible=True, which='both')
	plt.legend(["CNN","Uncertainty bands"])#, prop={'size': 16})
	plt.grid(True)
	plt.show()
	
def load_rawdata_csv(inputdata):
	X_data=[]
	with open(inputdata, newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		# next(spamreader)
		k=0
		for row in spamreader:
			if(row[0]==''):
				break
			else:
				x = np.array(row[:])
				x = x.astype(np.float)
				X_data.append(x)
				# print(x)
				# print(y)
				# input('test')
		# print(k)
	X_data=np.array(X_data)
	return X_data
