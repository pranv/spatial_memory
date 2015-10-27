import autograd.numpy as np
from autograd import grad
from layers import LSTM, Dense, softmax, softplus, sigmoid, tanh, ReLU


class Controller(object):
	def __init__(self, dmemory, daddress, nstates, dinput, doutput):
		self.layers = {}

		self.layers['INPUT'] = Dense(dinput, dinput)
		self.layers['PREVIOUS_READ'] = Dense(dmemory, dmemory)
		
		self.layers['CONTROL_KEY'] = LSTM(dinput + dmemory, nstates)
		
		self.layers['HASH'] = Dense(dmemory, daddress)
		
		self.layers['CONTENT_KEY_R'] = Dense(nstates, dmemory)
		self.layers['GATE_R'] = Dense(nstates, daddress)
		self.layers['LOCATION_R'] = Dense(nstates, daddress) 
		
		self.layers['CONTENT_KEY_W'] = Dense(nstates, dmemory)
		self.layers['GATE_W'] = Dense(nstates, daddress)
		self.layers['LOCATION_W'] = Dense(nstates, daddress) 
		self.layers['ERASE'] = Dense(nstates, dmemory)
		self.layers['ADD'] = Dense(nstates, dmemory) 
		self.layers['OUTPUT'] = Dense(nstates, doutput)

	def __call__(self, input, prev_read):
		layer = self.layers 	# alias
		I = layer['INPUT'](input) 
		P = layer['PREVIOUS_READ'](prev_read)
		C = layer['CONTROL_KEY'](np.concatenate([I, P]))

		content_r = layer['CONTENT_KEY_R'](C)
		content_w = layer['CONTENT_KEY_W'](C)

		loc_r = layer['LOCATION_R'](C)
		loc_w = layer['LOCATION_W'](C)

		g_r = sigmoid(layer['GATE_R'](C))
		g_w = sigmoid(layer['GATE_W'](C))

		address_r = loc_r * g_r + (1 - g_r) * layer['HASH'](content_r)
		address_w = loc_w * g_w + (1 - g_w) * layer['HASH'](content_w)

		erase = sigmoid(layer['ERASE'](C)) 
		add = layer['ADD'](C)

		output = sigmoid(layer['OUTPUT'](C))

		return address_r, address_w, erase, add, output

	def get_params(self):
		params = []
		for layer in self.layers:
			layer_params = self.layers[layer].get_params()
			params.append(layer_params)
		params = np.concatenate(params, axis=1)
		return params

	def set_params(self, params):
		for layer in self.layers:
			n = np.prod(self.layers[layer].W.shape)
			if self.layers[layer].__class__.__name__ == 'LSTM':
				n += 2 * np.prod(self.layers[layer].Y.shape)
			layer_params = params[-n:]
			self.layers[layer].set_params(layer_params)
			params = params[:-n]

	def clear(self):
		for layer in self.layers:
			self.layers[layer].clear()
