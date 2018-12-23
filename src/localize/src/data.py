#!/usr/bin/env python
import numpy as np
import scipy.stats
import sklearn.model_selection
from collections import Counter
import os 

class DataLoader:
	def __init__(self, file_name,num=6):
		self.pos = np.genfromtxt(file_name, usecols=(1,2), delimiter=",")
		macaddress = np.genfromtxt(file_name, usecols=(5,7,9,11,13,15,17,19,21),delimiter=",",dtype=str)
		unsorted_strength = np.genfromtxt(file_name, usecols=(6,8,10,12,14,16,18,20,22),delimiter=",")
		cat_strength = np.zeros((unsorted_strength.shape[0],unsorted_strength.shape[1]+1))
		cat_strength[:,:-1] = unsorted_strength		
		
		self.strength = np.empty([macaddress.shape[0],num])
		
		addresses = macaddress.flatten()
		#print("addresses",addresses)
		counter = Counter(addresses)
		list_tuples = counter.most_common(num)
		print("counter", list_tuples)
		self.addresses = [tuples[0] for tuples in list_tuples][:]

		for i in range(macaddress.shape[0]):
			index = self.bssid_sorted(self.addresses, macaddress[i], num)
			self.strength[i] = cat_strength[i][index]
		print("sorted strength",self.strength.shape)
		self.train_pos, self.test_pos, self.train_strength, self.test_strength = sklearn.model_selection.train_test_split(self.pos[:-20,], self.strength[:-20,])
		
	
	def get_user(self, bssid, rssi):
		cat_strength = np.zeros(rssi.shape[0]+1)
		cat_strength[:-1] = rssi
		index = self.bssid_sorted(self.addresses, bssid, self.strength.shape[1])
		print("index",index)
		strength = cat_strength[index]
		print("strength user", strength)
		return strength


	def bssid_sorted(self,address_list, macaddress, rssi):
		x = macaddress
		y = address_list[:rssi]

		index = np.argsort(x)
		sorted_x = x[index]
		sorted_index = np.searchsorted(sorted_x, y)

		yindex = np.take(index, sorted_index, mode="clip")
		mask = x[yindex] != y

		result = np.ma.array(yindex, mask=mask)
		result[result.mask] = -1
		return result.data

	def get_train(self):
		return self.train_pos, self.train_strength
		#return self.pos[:], self.strength[:]

	def get_test(self):
		return self.test_pos, self.test_strength
	def get_vali(self):
		return self.pos[45:51], self.strength[45:51]
if __name__ == '__main__':
	dir = os.path.dirname(os.path.realpath(__file__))
	file_name = dir + '/training_wifi.csv'
	d = DataLoader(file_name)
	pos,strength = d.get_train()
	print(pos.shape)
	print(strength.shape)
	print(strength[0])

