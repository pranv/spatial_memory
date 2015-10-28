import autograd.numpy as np

from controller import Controller
from memory import Memory
from layers import sigmoid

class SpatialMemoryMachine(object):
	def __init__(self, dmemory, daddress, nstates, dinput, doutput,
					init_units, create_memories, influence_threshold, sigma):

		self.memory = Memory(dmemory, daddress, init_units, create_memories, influence_threshold, sigma)
		self.controller = Controller(dmemory, daddress, nstates, dinput, doutput)
		self.doutput = doutput
		self.read0 = np.random.randn(dmemory)
		self.read = self.read0

	def __call__(self, inputs):
		sequence_length = inputs.shape[0]

		outputs = []
		for t in range(sequence_length):
			address, erase, add = self.controller(inputs[t], self.read)
			self.memory.commit(address, erase, add)
			output = self.read = self.memory.fetch(address)
			outputs.append(sigmoid(output).reshape(1, -1))

		return np.concatenate(outputs, axis=0)

	def loss(self, inputs, targets):
		inputs, targets = map(np.array, [inputs, targets])
		outputs = self(inputs)
		ep = 2e-23
		loss = -np.sum(targets * np.log2(outputs + ep) + (1 - targets) * np.log2(1 - outputs + ep))
		return loss + ep

	def clear(self):
		self.read = self.read0
		self.memory.clear()
		self.controller.clear()

	def get_params(self):
		params = self.controller.get_params()
		return np.concatenate([params, self.read0.flatten()], axis=1)

	def set_params(self, params):
		shape_r = self.read0.shape
		read0 = params[-np.prod(shape_r):]
		self.read0 = read0.reshape(shape_r)
		self.controller.set_params(params[:-np.prod(shape_r)])