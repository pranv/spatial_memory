import autograd.numpy as np

from layers import deccan

class Memory(object):
	def __init__(self, dmemory, daddress, scale, init_units, create_memories, influence_threshold, sigma):
		self.init_units = init_units
		self.create_memories = create_memories
		self.influence_threshold = influence_threshold
		self.sigma = sigma
		self.daddress = daddress
		self.dmemory = dmemory
		self.scale = scale
		self.clear()

	def activate(self, address):
		return deccan(self.locations, address * self.scale, self.sigma)

	def fetch(self, address):
		activations = self.activate(address)
		recall = activations * self.values
		read = np.sum(recall, axis=0)
		return read

	def commit(self, address, erase, add):
		activations = self.activate(address)

		if self.create_memories:
			if (np.sum(activations) < self.influence_threshold):
				self.create(address)
				activations = self.activate(address * self.scale)

		refresh = self.values * (1 - erase) + add 
		self.values = self.values * (1 - activations) + activations * refresh

	def create(self, address):
		self.values = np.concatenate([self.values, np.zeros((1, self.dmemory))])
		self.locations = np.concatenate([self.locations, address.reshape(1, -1)])

	def get_values(self):
		return self.values

	def get_locations(self):
		return self.locations

	def set_values(self, values):
		self.values = values

	def set_locations(self, locations):
		self.locations = locations

	def clear(self):
		self.values = np.random.random((self.init_units, self.dmemory))
		self.locations = np.ones((self.init_units, self.daddress)) * np.linspace(-1 * self.scale, 10 * self.scale, self.init_units).reshape(self.init_units, 1)
