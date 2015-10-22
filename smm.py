import autograd.numpy as np
from autograd import grad

from layers import *


class MemoryStore(object):
	def __init__(self, dmemory, daddress, write_threshold):
		self.values = np.zeros((1, dmemory))
		self.locations = np.zeros((1, daddress))
		self.write_threshold = write_threshold
		self.daddress = daddress
		self.dmemory = dmemory
		self.read = np.zeros(dmemory)

	def activate(self, address):
		# Gaussian
		sigma = 0.001
		mu = address
		coeff = 1 / (sigma * ((2 * 3.14) ** 0.5))
		norm = (self.locations - mu) ** 2
		fac = -np.sum(norm / (2 * (sigma ** 2)), axis=1, keepdims=True)
		return coeff * np.exp(fac)

	def fetch(self, address):
		activations = self.activate(address)
		recall = activations * self.values
		self.read = np.sum(recall, axis=0)

	def commit(self, address, erase, add):
		activations = self.activate(address)
		# create a new memory
		if np.sum(activations) < self.write_threshold:
			self.create(address)
			activations = self.activate(address)

		recall = activations * self.values
		refresh = self.values * (1 - erase) + add 
		self.values = self.values * (1 - recall) + recall * refresh

	def create(self, address):
		self.values = np.concatenate([self.values, np.zeros((1, self.dmemory))])
		self.locations = np.concatenate([self.locations, address.reshape(1, -1)])

	def get_values(self):
		return self.values

	def get_locations(self):
		return self.locations

	def get_read(self):
		return self.read

	def set_values(self, values):
		self.values = values

	def set_locations(self, locations):
		self.locations = locations

	def set_read(self, read):
		self.read = read

	def clear(self):
		self.values = np.zeros((1, self.dmemory))
		self.locations = np.zeros((1, self.daddress))


class SpatialMemoryMachine(object):
	def __init__(self, dmemory, daddress, nstates, dinput, doutput, write_threshold=1e-2):
		self.MEMORY = MemoryStore(dmemory, daddress, write_threshold)
		self.CONTROLLER = LSTM(dinput + dmemory, nstates)
		self.CONTENT_KEY = Dense(nstates, dmemory) 
		self.LOCATION = Dense(nstates, daddress) 
		self.GATE = Dense(nstates, daddress)
		self.ERASE = Dense(nstates, dmemory)
		self.ADD = Dense(nstates, dmemory)  
		self.OUTPUT = Dense(nstates, doutput)

		self.HASH = Dense(dmemory, daddress)

		self.dinput = dinput
		self.dmemory = dmemory
		self.daddress = daddress
		self.nstates = nstates

	def forward(self, input):
		V = np.concatenate([input, self.MEMORY.get_read()])
		H = self.CONTROLLER(V)
		
		content_key = self.CONTENT_KEY(H)
		location = self.LOCATION(H)

		gate = sigmoid(self.GATE(H))
		
		erase = sigmoid(self.ERASE(H))
		add = np.tanh(self.ADD(H))
		
		output = sigmoid(self.OUTPUT(H))
		
		content_address = self.HASH(content_key)
		address = (1 - gate) * content_address + gate * location
		
		self.MEMORY.fetch(address)
		self.MEMORY.commit(address, erase, add)

		#print '--' * 40
		#print 'content_key', content_key
		#print 'location', location
		#print 'gate', gate #, self.GATE(H) 
		#print 'erase', erase
		#print 'add', add
		#print 'output', output
		return output

	def get_params(self):
		params = {}

		params['CONTROLLER_W'] = self.CONTROLLER.get_params() 
		params['CONTROLLER_c'] = self.CONTROLLER.get_prev_c()
		params['CONTROLLER_Y'] = self.CONTROLLER.get_prev_Y()

		params['CONTENT_KEY_W'] = self.CONTENT_KEY.get_params()
		params['LOCATION_W'] = self.LOCATION.get_params()
		params['GATE_W'] = self.GATE.get_params()
		params['ERASE_W'] = self.ERASE.get_params()
		params['ADD_W'] = self.ADD.get_params()
		params['OUTPUT_W'] = self.OUTPUT.get_params()
		params['HASH_W'] = self.HASH.get_params()

		return params

	def set_params(self, params):
		self.CONTROLLER.set_params(params['CONTROLLER_W']) 
		self.CONTROLLER.set_prev_c(params['CONTROLLER_c'])
		self.CONTROLLER.set_prev_Y(params['CONTROLLER_Y'])
		
		self.CONTENT_KEY.set_params(params['CONTENT_KEY_W'])
		self.LOCATION.set_params(params['LOCATION_W'])
		self.GATE.set_params(params['GATE_W'])
		self.ERASE.set_params(params['ERASE_W'])
		self.ADD.set_params(params['ADD_W'])
		self.OUTPUT.set_params(params['OUTPUT_W'])
		self.HASH.set_params(params['HASH_W'])

	def clear(self):
		self.MEMORY.clear()