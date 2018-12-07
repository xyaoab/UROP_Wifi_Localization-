import GPy
import numpy as np
import scipy.stats
from data import DataLoader

class Model:
	def __init__(self, train_pos, train_strength):
		kernel = GPy.kern.RBF(input_dim=train_pos.shape[1],variance=1.0, lengthscale=None)
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

	def save(self):
		for i in range(self.rssi):
			np.save('./static_model/model_'+str(i)+'_save.npy', self.m[i].param_array)
			#self.m[i]to_dict()
	def grid(self, amin, amax):
		xg1 =np.linspace(amin[0],amax[1])
		xg2 =np.linspace(amin[0],amax[1])
		self.test_X = np.zeros((xg1.size * xg2.size,2))
		for i,x1 in enumerate(xg1):
		    for j,x2 in enumerate(xg2):
		        self.test_X[i+xg1.size*j,:] = [x1,x2]

	def predict(self, test_strength):
		mean = []
		variance = []
		for i in range(self.rssi):
			mean.append(self.m[i].predict(self.test_X)[0])
			variance.append(self.m[i].predict(self.test_X)[1])

		prob = np.ndarray((self.rssi,self.test_X.shape[0]))

		for j in range(self.rssi):
				prob[j]= scipy.stats.norm(mean[j], variance[j]**2**0.25).pdf(test_strength[j]).reshape(1,-1)
		return prob


	def bayes_filter(self, prob, test_pos, test_strength):
		self.belief= prob * self.predict(test_strength)
		self.belief = self.belief / self.belief.sum(axis=0)
		print(self.belief.shape)
		print("average", np.average(self.belief, axis=0).shape)

		self.predict_pos = self.test_X[np.argmax(np.average(self.belief, axis=0))]
		print("predict+pos", self.predict_pos)
		sse = ((self.predict_pos - test_pos)**2).sum()
		print("sse", sse)
		


if __name__ == '__main__':
	file_name='./training_wifi.csv'
	d = DataLoader(file_name)
	train_pos = d.get_train()[0]
	train_strength = d.get_train()[1]
	model_list = Model(train_pos,train_strength)
	model_list.train()
	#model_list.save()
	model_list.grid(np.amin(train_pos,axis=0),np.amax(train_pos,axis=1))
	test_pos =  d.get_vali()[0]
	test_strength =  d.get_vali()[1]
	print ("test_strength",test_strength)
	for i in range(test_pos.shape[0]):
		model_list.bayes_filter(model_list.belief,test_pos[i], test_strength[i])
	#model_list.predict(d.get_test()[0], d.get_test()[1])

