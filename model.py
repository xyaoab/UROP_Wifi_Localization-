import GPy
import numpy as np
import scipy.stats
from data import DataLoader

class Model:
	def __init__(self, train_pos, train_strength):
		kernel = GPy.kern.RBF(input_dim=train_pos.shape[1],variance=1.0, lengthscale=None)
		self.rssi = train_strength.shape[1]
		self.m = []
		for i in range(self.rssi):
			self.m.append(GPy.models.GPRegression(train_pos, train_strength[:,i].reshape(-1,1),kernel))
	def train(self):
		for i in range(self.rssi):
			self.m[i].constrain_positive('.*rbf_variance')
			self.m[i].Gaussian_noise.constrain_positive() 
			self.m[i].randomize()
			self.m[i].optimize(messages=True)

	def save(self):
		for i in range(self.rssi):
			np.save('./static_model/model_'+str(i)+'_save.npy', self.m[i].param_array)
			#self.m[i]to_dict()
	def grid(self):
		xg1 =np.linspace(-2,1)
		xg2 =np.linspace(-2,1)
		self.test_X = np.zeros((xg1.size * xg2.size,2))
		for i,x1 in enumerate(xg1):
		    for j,x2 in enumerate(xg2):
		        self.test_X[i+xg1.size*j,:] = [x1,x2]

	def predict(self, test_pos, test_strength):
		self.mean = []
		self.variance = []
		for i in range(self.rssi):
			self.mean.append(self.m[i].predict(self.test_X)[0])
			self.variance.append(self.m[i].predict(self.test_X)[1])

		self.prob = np.ndarray((test_strength.shape[0], self.test_X.shape[0],self.rssi))

		for j in range(self.rssi):
			for i in range(self.prob.shape[0]):
				self.prob[i] = scipy.stats.norm(self.mean[j], self.variance[j]**2**0.25).pdf(test_strength[i,j].reshape(1,-1))
		self.predict_pos = self.test_X[np.argmax(np.average(self.prob, axis=2), axis=1)]
		sse = ((self.predict_pos - test_pos)**2).sum()
		print(sse)



if __name__ == '__main__':
	file_name='./hehe.csv'
	d = DataLoader(file_name)
	train_pos = d.get_train()[0]
	train_strength = d.get_train()[1]
	model_list = Model(train_pos,train_strength)
	model_list.train()
	model_list.save()
	model_list.grid()
	model_list.predict(d.get_test()[0], d.get_test()[1])

