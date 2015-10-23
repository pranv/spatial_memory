import autograd.numpy as np
from autograd import grad

from layers import LSTM, Dense, softmax, softplus, sigmoid

class MemoryStore(object):
	def __init__(self, dmemory, daddress, write_threshold=1e-3, sigma=0.01):
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
	def __init__(self, dmemory, daddress, nstates, dinput, doutput, write_threshold, sigma):
		self.MEMORY = MemoryStore(dmemory, daddress, write_threshold, sigma)
		self.CONTROLLER = LSTM(dinput + dmemory, nstates)
		
		self.CONTENT_KEY_R = Dense(nstates, dmemory) 
		self.LOCATION_R = Dense(nstates, daddress) 
		self.GATE_R = Dense(nstates, 1)
		self.ERASE_R = Dense(nstates, dmemory)
		self.ADD_R = Dense(nstates, dmemory) 
		self.HASH_R = Dense(dmemory, daddress)

		self.CONTENT_KEY_W = Dense(nstates, dmemory) 
		self.LOCATION_W = Dense(nstates, daddress) 
		self.GATE_W = Dense(nstates, 1)
		self.ERASE_W = Dense(nstates, dmemory)
		self.ADD_W = Dense(nstates, dmemory) 
		self.HASH_W = Dense(dmemory, daddress)

		self.OUTPUT = Dense(nstates, doutput)

		self.dinput = dinput
		self.dmemory = dmemory
		self.daddress = daddress
		self.nstates = nstates

	def forward(self, input, verbose=False):
		prev_read = self.MEMORY.get_read()
		V = np.concatenate([input, prev_read])
		H = self.CONTROLLER(V)
		
		content_key_r = np.tanh(self.CONTENT_KEY_R(H))
		location_r = np.tanh(self.LOCATION_R(H))
		gate_r = sigmoid(self.GATE_R(H))
		content_address_r = np.tanh(self.HASH_R(content_key_r))
		address_r = (1 - gate_r) * content_address_r + gate_r * location_r
		self.MEMORY.fetch(address_r)

		content_key_w = np.tanh(self.CONTENT_KEY_W(H))
		location_w = np.tanh(self.LOCATION_W(H))
		gate_w = sigmoid(self.GATE_W(H))
		erase = sigmoid(self.ERASE_W(H))
		add = np.tanh(self.ADD_W(H))
		content_address_w = np.tanh(self.HASH_W(content_key_w))
		address_w = (1 - gate_w) * content_address_w + gate_w * location_w
		self.MEMORY.commit(address_w, erase, add)

		#if verbose:
		#	print 'prev_read: \t\t', prev_read
		#	print 'hidden: ', H
		#	print 'content_key_r: \t\t', content_key_r
		#	print 'location_r: \t\t', location_r
		#	print 'gate_r: \t\t', gate_r #, self.GATE(H)
		#	print 'content_address_r: \t\t', content_address_r
		#	print 'address_r: \t\t', address_r
		#	print '.',
		#	print 'content_key_w: \t\t', content_key_w
		#	print 'location_w: \t\t', location_w
		#	print 'gate_w: \t\t', gate_w #, self.GATE(H) 
		#	print 'content_address_r: \t\t', content_address_r
		#	print 'address_w: \t\t', address_w
		#	print 'erase_w: \t\t', erase
		#	print 'add_w: \t\t', add
		#	print '------' * 10	

		output = sigmoid(self.OUTPUT(H))

		return output

	def get_params(self):
		params = {}

		params['CONTROLLER'] = self.CONTROLLER.get_params() 
		params['CONTROLLER_c'] = self.CONTROLLER.get_prev_c()
		params['CONTROLLER_Y'] = self.CONTROLLER.get_prev_Y()

		params['CONTENT_KEY_W'] = self.CONTENT_KEY_W.get_params()
		params['LOCATION_W'] = self.LOCATION_W.get_params()
		params['GATE_W'] = self.GATE_W.get_params()
		params['ERASE_W'] = self.ERASE_W.get_params()
		params['ADD_W'] = self.ADD_W.get_params()
		params['HASH_W'] = self.HASH_W.get_params()

		params['CONTENT_KEY_R'] = self.CONTENT_KEY_R.get_params()
		params['LOCATION_R'] = self.LOCATION_R.get_params()
		params['GATE_R'] = self.GATE_R.get_params()
		params['ERASE_R'] = self.ERASE_R.get_params()
		params['ADD_R'] = self.ADD_R.get_params()
		params['HASH_R'] = self.HASH_R.get_params()

		params['OUTPUT'] = self.OUTPUT.get_params()

		return params

	def set_params(self, params):
		self.CONTROLLER.set_params(params['CONTROLLER']) 
		self.CONTROLLER.set_prev_c(params['CONTROLLER_c'])
		self.CONTROLLER.set_prev_Y(params['CONTROLLER_Y'])
		
		self.CONTENT_KEY_W.set_params(params['CONTENT_KEY_W'])
		self.LOCATION_W.set_params(params['LOCATION_W'])
		self.GATE_W.set_params(params['GATE_W'])
		self.ERASE_W.set_params(params['ERASE_W'])
		self.ADD_W.set_params(params['ADD_W'])
		self.HASH_W.set_params(params['HASH_W'])

		self.CONTENT_KEY_R.set_params(params['CONTENT_KEY_R'])
		self.LOCATION_R.set_params(params['LOCATION_R'])
		self.GATE_R.set_params(params['GATE_R'])
		self.ERASE_R.set_params(params['ERASE_R'])
		self.ADD_R.set_params(params['ADD_R'])
		self.HASH_R.set_params(params['HASH_R'])

		self.OUTPUT.set_params(params['OUTPUT'])

	def clear(self):
		self.MEMORY.clear()