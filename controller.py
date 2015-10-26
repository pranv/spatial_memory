import autograd.numpy as np
from autograd import grad
from layers import LSTM, Dense, softmax, softplus, sigmoid, tanh


class Controller(object):
	def __init__(self, dmemory, daddress, nstates, dinput, doutput):
		self.layers = {}
		self.layers['INPUT'] = Dense(dinput, nstates)
		self.layers['PREVIOUS_READ'] = Dense(dmemory, nstates)
		
		self.layers['CONTROL_KEY'] = LSTM(nstates, nstates)
		
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

		V = layer['INPUT'](input) + layer['PREVIOUS_READ'](prev_read)
		C = tanh(layer['CONTROL_KEY'](V))

		content_r = layer['CONTENT_KEY_R'](C)
		content_w = layer['CONTENT_KEY_W'](C)

		loc_r = layer['LOCATION_R'](C)
		loc_w = layer['LOCATION_W'](C)

		g_r = layer['GATE_R'](C)
		g_w = layer['GATE_W'](C)

		address_r = g_r * loc_r + (1 - g_r) * layer['HASH'](content_r)
		address_w = g_w * loc_w + (1 - g_w) * layer['HASH'](content_w)

		erase = layer['ERASE'](C)
		add = layer['ADD'](C)

		output = sigmoid(layer['OUTPUT'](C))

		return address_r, address_w, erase, add, output

	def get_params(self):
		params = {}
		for layer in self.layers:
			layer_params = self.layers[layer].get_params()
			params[layer + '_params'] = layer_params[0]
			if self.layers[layer].__class__.__name__ == 'LSTM':
				params[layer + '_c'] = layer_params[1]
				params[layer + '_Y'] = layer_params[2]
		return params

	def set_params(self, params):
		for layer in self.layers:
			layer_params = (params[layer + '_params'], )
			if self.layers[layer].__class__.__name__ == 'LSTM':
				layer_params += (params[layer + '_c'], )
				layer_params += (params[layer + '_Y'], )
			self.layers[layer].set_params(layer_params)

	def clear(self):
		for layer in self.layers:
			if self.layers[layer].__class__.__name__ == 'LSTM':
				self.layers[layer].clear()
