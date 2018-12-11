#!/usr/bin/env python
import GPy
import numpy as np
import scipy.stats
from data import DataLoader
import os
from GPy.util.linalg import dtrtrs



class Model:
	def __init__(self, train_pos, train_strength):
		kernel = GPy.kern.RBF(input_dim=train_pos.shape[1])
		self.belief = 1.
		self.rssi = train_strength.shape[1]
		self.m = []
		for i in range(self.rssi):
			self.m.append(GPy.models.GPRegression(train_pos, train_strength[:,i].reshape(-1,1),kernel))
	def train(self):
		for i in range(self.rssi):
			self.m[i].constrain_positive('')
			self.m[i].randomize()
			self.m[i].optimize(messages=True)

	def save(self, file_path):
		for i in range(self.rssi):
			np.save( file_path + '/static_model/model_'+str(i)+'_save.npy', self.m[i].param_array)
			#self.m[i]to_dict()
	def grid(self, amin, amax):
		xg1 =np.linspace(amin[0],amax[1],40)
		xg2 =np.linspace(amin[0],amax[1],40)
		self.test_X = np.zeros((xg1.size * xg2.size,2))
		for i,x1 in enumerate(xg1):
		    for j,x2 in enumerate(xg2):
		        self.test_X[i+xg1.size*j,:] = [x1,x2]

	def predict(self, test_strength):
		mean = []
		variance = []
		for i in range(self.rssi):
			mean.append(self.m[i].predict(self.test_X)[0])
			## try to solve the negative variance problem caused by the sparsity of the data
			#cov = self.m[i].kern.K(self.m[i].X)+np.eye(self.m[i].X.shape[0])*self.m[i].likelihood.variance.values
			#kfs = self.m[i].kern.K(self.m[i].X, self.test_X)
			#kss = self.m[i].kern.Kdiag(self.test_X)
			#var = kss -  (kfs*GPy.util.linalg.dpotrs(self.m[i].posterior.woodbury_chol, kfs)[0]).sum(0)
			#print ("min",var.min())
			#print("shape",var.shape)
			#print("variance", var)
			#variance.append(var)
			variance.append(self.m[i].predict(self.test_X)[1])

		prob = np.ndarray((self.rssi,self.test_X.shape[0]))

		for j in range(self.rssi):
				prob[j]= scipy.stats.norm(mean[j], variance[j] **.5).pdf(test_strength[j]).reshape(1,-1)
		return prob


	def bayes_filter(self, prob, test_strength,test_pos=None):
		self.belief= prob * self.predict(test_strength)
		self.belief = self.belief / self.belief.sum(axis=0)
		print(self.belief)
		#print("average", np.cumprod(self.belief, axis=0).shape)

		#self.predict_pos = self.test_X[np.argmax(np.cumprod(self.belief, axis=0)[-1]  ** (1/self.rssi))]
		self.predict_pos = self.test_X[np.argmax(np.average(self.belief, axis=0))]
		print("predict+pos", self.predict_pos)
		if test_pos is not None:
			sse = ((self.predict_pos - test_pos)**2).sum()
			print("sse", sse)
		


if __name__ == '__main__':
	dir = os.path.dirname(os.path.realpath(__file__))
	file_name = dir + '/training_wifi.csv'
	d = DataLoader(file_name)
	train_pos = d.get_train()[0]
	train_strength = d.get_train()[1]
	model_list = Model(train_pos,train_strength)
	model_list.train()
	#model_list.save(dir)
	model_list.grid(np.amin(train_pos,axis=0),np.amax(train_pos,axis=1))
	test_pos =  d.get_vali()[0]
	test_strength =  d.get_vali()[1]
	print ("test_strength",test_strength)
	for i in range(test_pos.shape[0]):
		model_list.bayes_filter(model_list.belief, test_strength[i],test_pos[i])

