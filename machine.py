import autograd.numpy as np

from controller import Controller
from memory import Memory

class SpatialMemoryMachine(object):
	def __init__(self, dmemory, daddress, nstates, dinput, doutput, 
					init_units, create_memories, influence_threshold, sigma):

		self.memory = Memory(dmemory, daddress, init_units, create_memories, influence_threshold, sigma)
		self.controller = Controller(dmemory, daddress, nstates, dinput, doutput)
		self.doutput = doutput
		self.read0 = np.random.randn(dmemory)

	def __call__(self, inputs):
		sequence_length = inputs.shape[0]
		self.read = self.read0

		outputs = []
		for t in range(sequence_length):
			address_r, address_w, erase, add, output = self.controller(inputs[t], self.read)
			#print address_r#.value, address_w.value, erase.value, add.value, output.value
			self.memory.commit(address_w, erase, add)
			self.read = self.memory.fetch(address_r)
			outputs.append(output)

		return np.array(outputs)

	def loss(self, inputs, targets):
		inputs, targets = map(np.array, [inputs, targets])
		outputs = self.__call__(inputs)
		ep = 2e-23
		loss = -np.sum(targets * np.log2(outputs + ep) + (1 - targets) * np.log2(1 - outputs + ep))
		return loss

	def clear(self):
		self.read = self.read0
		self.memory.clear()
		self.controller.clear()

	def get_params(self):
		params = self.controller.get_params()
		params['read0'] = self.read0
		return params

	def set_params(self, params):
		self.read0 = params['read0']
		self.controller.set_params(params)