import numpy as np


def sigmoid(X):
	return 1.0 / (1.0 + np.exp(-X))


class MemoryStore(object):
	def __init__(self, dmemory, daddress, write_threshold, sigma):
		self.values = np.zeros((1, dmemory))
		self.locations = np.zeros((1, daddress))
		self.write_threshold = write_threshold
		self.sigma = sigma
		self.daddress = daddress
		self.dmemory = dmemory
		self.read = np.zeros(dmemory)

	def activate(self, address):
		# Gaussian
		sigma = self.sigma
		mu = address
		norm = (self.locations - mu) ** 2
		fac = -np.sum(norm / (2 * (sigma ** 2)), axis=1, keepdims=True)
		return np.exp(fac)

	def fetch(self, address):
		activations = self.activate(address)
		recall = activations * self.values
		self.read = np.sum(recall, axis=0)
		return self.read

	def commit(self, address, erase, add):
		activations = self.activate(address)
		# create a new memory
		if np.sum(activations) < self.write_threshold:
			self.create(address)
			activations = self.activate(address)

		refresh = self.values * (1 - erase) + add 
		self.values = self.values * (1 - activations) + activations * refresh

	def create(self, address):
		self.values = np.concatenate([self.values, np.zeros((1, self.dmemory))])
		self.locations = np.concatenate([self.locations, address.reshape(1, -1)])


MEM = MemoryStore(dmemory=10, daddress=1, write_threshold=1e-3, sigma=0.01)
address = np.array([[0.2]])
memory = np.ones((1, 10))
MEM.commit(address, erase=0, add=memory)

print MEM.values, MEM.locations

MEM.commit(address, erase=0, add=memory)

print MEM.values, MEM.locations


MEM.commit(address+0.001, erase=0, add=memory)

print MEM.values, MEM.locations

print 'memory activation: ', MEM.activate(address)

locations = np.linspace(-2, 2, 10000)
activations = np.array([MEM.activate(locations[i]) for i in range(10000)])
print activations.shape
import matplotlib.pyplot as plt
plt.ion()
plt.plot(locations, activations.sum(axis=1))

raw_input()