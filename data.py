import GPy
import csv
import numpy as np
import scipy.stats
import sklearn.model_selection

class DataLoader:
	def __init__(self, file_name):
		all_time = np.genfromtxt(file_name,usecols=(0), delimiter=",",dtype=None)
		self.pos = np.genfromtxt(file_name, usecols=(1,2), delimiter=",")
		macaddress = np.genfromtxt(file_name, usecols=(5,7,9),delimiter=",",dtype=str)
		self.strength = np.genfromtxt(file_name, usecols=(6,8,10),delimiter=",")
		timestamp, indices = np.unique(all_time, return_index=True)
		self.train_pos, self.test_pos, self.train_strength, self.test_strength = sklearn.model_selection.train_test_split(self.pos, self.strength)

	def get_train(self):
		return self.train_pos, self.train_strength

	def get_test(self):
		return self.test_pos, self.test_strength


if __name__ == '__main__':
	file_name='./hehe.csv'
	d = DataLoader(file_name)
	print(d.get_train()[0].shape)
	print(d.get_train()[1].shape)