import autograd.numpy as np
from autograd import grad

from layers import LSTM, Dense, softmax, softplus, sigmoid

class MemoryStore(object):
	def __init__(self, dmemory, daddress, write_threshold=0.5, sigma=0.05):
		self.values = np.zeros((25, dmemory))
		self.locations = np.ones((25, daddress)) * np.linspace(-1, 1, 25).reshape(25, 1)
		self.write_threshold = write_threshold
		self.sigma = sigma
		self.daddress = daddress
		self.dmemory = dmemory
		self.read = np.zeros(dmemory)

	def activate(self, address):
		norm = np.sum((self.locations - address) ** 2, axis=1, keepdims=True)
		return np.exp(-norm / (2 * (self.sigma ** 2))) + 1 / (norm + 0.999)

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
		self.values = np.zeros((25, self.dmemory))
		self.locations = np.ones((25, self.daddress)) * np.linspace(-1.0, 1.0, 25).reshape(25, 1)
		

class SpatialMemoryMachine(object):
	def __init__(self, dmemory, daddress, nstates, dinput, doutput, write_threshold=1e-3, sigma=0.01):
		self.MEMORY = MemoryStore(dmemory, daddress, write_threshold, sigma)
		self.CONTROLLER = LSTM(dinput, nstates)
		self.PREV_READ = LSTM(dmemory, nstates)
		self.MASTER = LSTM(nstates, nstates)
		
		self.CONTENT_KEY_R = LSTM(nstates, dmemory)
		self.GATE_R = LSTM(nstates, daddress)
		self.HASH_R= LSTM(dmemory, daddress)
		self.LOCATION_R = LSTM(nstates, daddress) 

		self.CONTENT_KEY_W = LSTM(nstates, dmemory)
		self.GATE_W = LSTM(nstates, daddress)
		self.HASH_W = LSTM(dmemory, daddress)
		self.LOCATION_W = LSTM(nstates, daddress) 
		self.ERASE = LSTM(nstates, dmemory)
		self.ADD = LSTM(nstates, dmemory) 

		self.OUTPUT = LSTM(nstates, doutput)

		self.dinput = dinput
		self.dmemory = dmemory
		self.daddress = daddress
		self.nstates = nstates

	def forward(self, input, verbose=False):
		H2 = self.PREV_READ(self.MEMORY.get_read())
		H1 = np.tanh(self.CONTROLLER(np.array(input)))
		h = H1 + H2
		H =	np.tanh(self.MASTER(h))

		location_r = np.tanh(self.LOCATION_R(H))
		content_key_r = softplus(self.CONTENT_KEY_R(H))
		content_address_r = np.tanh(self.HASH_R(content_key_r))
		gate_r = sigmoid(self.GATE_R(H))
		address_r = (1 - gate_r) * content_address_r + gate_r * location_r
		self.MEMORY.fetch(address_r)

		location_w = np.tanh(self.LOCATION_W(H))
		content_key_w = softplus(self.CONTENT_KEY_W(H))
		content_address_w = np.tanh(self.HASH_W(content_key_w))
		gate_w = sigmoid(self.GATE_W(H))
		address_w = (1 - gate_w) * content_address_w + gate_w * location_w
		erase = sigmoid(self.ERASE(H))
		add = np.tanh(self.ADD(H))
		self.MEMORY.commit(address_w, erase, add)

		output = sigmoid(self.OUTPUT(H))

		return output

	def get_params(self):
		params = {}

		params['CONTROLLER'] = self.CONTROLLER.get_params() 
		params['CONTROLLER_c'] = self.CONTROLLER.get_prev_c()
		params['CONTROLLER_Y'] = self.CONTROLLER.get_prev_Y()

		params['PREV_READ_c'] = self.PREV_READ.get_prev_c()
		params['PREV_READ_Y'] = self.PREV_READ.get_prev_Y()
		params['PREV_READ'] = self.PREV_READ.get_params() 

		params['MASTER'] = self.MASTER.get_params() 
		params['MASTER_c'] = self.MASTER.get_prev_c()
		params['MASTER_Y'] = self.MASTER.get_prev_Y()

		params['CONTENT_KEY_R'] = self.CONTENT_KEY_R.get_params()
		params['CONTENT_KEY_R_c'] = self.CONTENT_KEY_R.get_prev_c()
		params['CONTENT_KEY_R_Y'] = self.CONTENT_KEY_R.get_prev_Y()
		
		params['GATE_R'] = self.GATE_R.get_params()
		params['GATE_R_c'] = self.GATE_R.get_prev_c()
		params['GATE_R_Y'] = self.GATE_R.get_prev_Y()
		
		params['HASH_R'] = self.HASH_R.get_params()
		params['HASH_R_c'] = self.HASH_R.get_prev_c()
		params['HASH_R_Y'] = self.HASH_R.get_prev_Y()

		params['LOCATION_R'] = self.LOCATION_R.get_params()
		params['LOCATION_R_c'] = self.LOCATION_R.get_prev_c()
		params['LOCATION_R_Y'] = self.LOCATION_R.get_prev_Y()
		
		params['CONTENT_KEY_W'] = self.CONTENT_KEY_W.get_params()
		params['CONTENT_KEY_W_c'] = self.CONTENT_KEY_W.get_prev_c()
		params['CONTENT_KEY_W_Y'] = self.CONTENT_KEY_W.get_prev_Y()
		
		params['GATE_W'] = self.GATE_W.get_params()
		params['GATE_W_c'] = self.GATE_W.get_prev_c()
		params['GATE_W_Y'] = self.GATE_W.get_prev_Y()
		
		params['HASH_W'] = self.HASH_W.get_params()
		params['HASH_W_c'] = self.HASH_W.get_prev_c()
		params['HASH_W_Y'] = self.HASH_W.get_prev_Y()
		
		params['LOCATION_W'] = self.LOCATION_W.get_params()
		params['LOCATION_W_c'] =self.LOCATION_W.get_prev_c()
		params['LOCATION_W_Y'] = self.LOCATION_W.get_prev_Y()
		
		params['ERASE'] = self.ERASE.get_params()
		params['ERASE_c'] = self.ERASE.get_prev_c()
		params['ERASE_Y'] = self.ERASE.get_prev_Y()
		
		params['ADD'] = self.ADD.get_params()
		params['ADD_c'] = self.ADD.get_prev_c()
		params['ADD_Y'] = self.ADD.get_prev_Y()

		params['OUTPUT'] = self.OUTPUT.get_params()
		params['OUTPUT_c'] = self.OUTPUT.get_prev_c()
		params['OUTPUT_Y'] = self.OUTPUT.get_prev_Y()

		return params

	def set_params(self, params):
		self.CONTROLLER.set_params(params['CONTROLLER']) 
		self.CONTROLLER.set_prev_c(params['CONTROLLER_c'])
		self.CONTROLLER.set_prev_Y(params['CONTROLLER_Y'])

		self.PREV_READ.set_params(params['PREV_READ']) 
		self.PREV_READ.set_prev_c(params['PREV_READ_c'])
		self.PREV_READ.set_prev_Y(params['PREV_READ_Y'])

		self.MASTER.set_params(params['MASTER']) 
		self.MASTER.set_prev_c(params['MASTER_c'])
		self.MASTER.set_prev_Y(params['MASTER_Y'])

		self.CONTENT_KEY_R.set_params(params['CONTENT_KEY_R'])
		self.CONTENT_KEY_R.set_prev_c(params['CONTENT_KEY_R_c'])
		self.CONTENT_KEY_R.set_prev_Y(params['CONTENT_KEY_R_Y'])
		
		self.GATE_R.set_params(params['GATE_R'])
		self.GATE_R.set_prev_c(params['GATE_R_c'])
		self.GATE_R.set_prev_Y(params['GATE_R_Y'])
		
		self.HASH_R.set_params(params['HASH_R'])
		self.HASH_R.set_prev_c(params['HASH_R_c'])
		self.HASH_R.set_prev_Y(params['HASH_R_Y'])

		self.LOCATION_R.set_params(params['LOCATION_R'])
		self.LOCATION_R.set_prev_c(params['LOCATION_R_c'])
		self.LOCATION_R.set_prev_Y(params['LOCATION_R_Y'])
		
		self.CONTENT_KEY_W.set_params(params['CONTENT_KEY_W'])
		self.CONTENT_KEY_W.set_prev_c(params['CONTENT_KEY_W_c'])
		self.CONTENT_KEY_W.set_prev_Y(params['CONTENT_KEY_W_Y'])
		
		self.GATE_W.set_params(params['GATE_W'])
		self.GATE_W.set_prev_c(params['GATE_W_c'])
		self.GATE_W.set_prev_Y(params['GATE_W_Y'])
		
		self.HASH_W.set_params(params['HASH_W'])
		self.HASH_W.set_prev_c(params['HASH_W_c'])
		self.HASH_W.set_prev_Y(params['HASH_W_Y'])
		
		self.LOCATION_W.set_params(params['LOCATION_W'])
		self.LOCATION_W.set_prev_c(params['LOCATION_W_c'])
		self.LOCATION_W.set_prev_Y(params['LOCATION_W_Y'])
		
		self.ERASE.set_params(params['ERASE'])
		self.ERASE.set_prev_c(params['ERASE_c'])
		self.ERASE.set_prev_Y(params['ERASE_Y'])
		
		self.ADD.set_params(params['ADD']  )
		self.ADD.set_prev_c(params['ADD_c'])
		self.ADD.set_prev_Y(params['ADD_Y'])

		self.OUTPUT.set_params(params['OUTPUT'])
		self.OUTPUT.set_prev_c(params['OUTPUT_c'])
		self.OUTPUT.set_prev_Y(params['OUTPUT_Y'])

	def clear(self):
		self.MEMORY.clear()