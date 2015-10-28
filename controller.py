import autograd.numpy as np
from autograd import grad
from layers import LSTM, Dense, softmax, softplus, sigmoid, tanh, ReLU


class Controller(object):
	def __init__(self, dmemory, daddress, nstates, dinput, doutput):
		self.layers = {}
		self.layers['CONTROL_KEY'] = LSTM(dinput + dmemory, nstates, 1000.0)
		self.layers['LOCATION'] = Dense(nstates, daddress)  
		self.layers['ERASE'] = Dense(nstates, dmemory)
		self.layers['ADD'] = Dense(nstates, dmemory) 


	def __call__(self, input, prev_read):
		layer = self.layers 	# alias
		C = layer['CONTROL_KEY'](np.concatenate([input, prev_read]))

		address = tanh(layer['LOCATION'](C))

		erase = 1 #sigmoid(layer['ERASE'](C)) 
		add = input[:-2] #tanh(layer['ADD'](C))

		return address, erase, add

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
