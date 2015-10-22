import autograd.numpy as np
from autograd import grad

from layers import *


class MemoryStore(object):
	def __init__(self, daddress, dmemory, write_threshold):
		self.values = np.zeros((1, dmemory))
		self.addresses = np.zeros((1, daddress))
		self.write_threshold = write_threshold
		self.daddress = daddress
		self.dmemory = dmemory

	def RBF(location, addresses):
		norm = (addresses - location) ** 2
		return 1e-3 / np.sum(norm, axis=1, keepdims=True)

	def read(self, address):
		activations = RBF(address, self.addresses)
		memory_activations = activations * self.values
		return np.sum(memory_activations, axis=0)

	def write(self, address, edits, erase, add):
		activations = RBF(address, self.addresses)
		if np.sum(activations) < self.write_threshold:
			self.create(address, edits, erase, add)
		else:
			memory_activations = activations * self.values
			new_memory = self.values * (1 - erase) + add * edits 
			self.values = self.values * (1 - memory_activations) + memory_activations * new_memory

	def create(self, address, edits, erase, add):
		new_memory = np.zeros((1, self.dmemory)) * (1 - erase) + add * edits 
		self.values = np.concatenate([self.values, new_memory])
		self.addresses = np.concatenate([self.addresses, address.reshape(1, -1)]) 



class SpatialMemoryMachine(object):
	def __init__(self, dinput, memory_size, daddress):
		self.memory = MemoryStore(daddress, memory_size, 1e2)
		self.controller = LSTM(dinput + memory_size, daddress + memory_size * 4 + 1 + dinput)
		self.hash = Dense(memory_size, daddress)
		self.output = Dense(dinput, dinput)

		self.dinput = dinput
		self.memory_size = memory_size
		self.daddress = daddress

	def forward(self, inputs):
		input = np.concatenate([inputs, self.read])

		commands = self.controller(input)
		content_key, location, retrieval_gate, erase, add, edits, output = self.unwrap(commands)
		
		content_address = self.hash(content_key)
		address = retrieval_gate * content_address + (1 - retrieval_gate) * location
		
		self.read = self.memory.read(address)
		self.memory.write(address, edits, softmax(erase), softmax(add))

		Y = softmax(output)

		return Y

	def loss(self, X, targets):
		self.read = np.zeros(self.memory_size)
		loss = 0
		T = X.shape[0]
		self.controller.forget()

		for t in range(T):
			Y = self.forward(X[t])
			one = np.ones(Y.shape)
	        target = targets[t]

	        ep = 2**-23 # to prevent log(0)
	        loss += target * np.log2(Y + ep) + (one - target) * np.log2(one - Y + ep)
		
		return -loss

	def unwrap(self, commands):
		memory_size = self.memory_size
		dinput = self.dinput
		daddress = self.daddress

		content_key = commands[:memory_size]
		commands = commands[memory_size:]		
		
		location = commands[:daddress]
		commands = commands[daddress:]
		
		retrieval_gate = commands[:1]
		commands = commands[1:]

		output = commands[3 * memory_size:]
		
		erase, add, edits = np.split(commands[:3 * memory_size], 3)
		
		return content_key, location, retrieval_gate, erase, add, edits, output
