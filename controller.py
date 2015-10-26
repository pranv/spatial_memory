import autograd.numpy as np
from autograd import grad
from layers import LSTM, Dense, softmax, softplus, sigmoid, tanh, ReLU


class Controller(object):
	def __init__(self, dmemory, daddress, nstates, dinput, doutput):
		self.layers = {}
		self.layers['INPUT'] = Dense(dinput, nstates)
		self.layers['PREVIOUS_READ'] = Dense(dmemory, nstates)
		self.layers['CONTROL_KEY'] = LSTM(nstates, nstates + 4 * (daddress + dmemory))
		self.layers['HASH'] = Dense(dmemory, daddress)
		self.layers['OUTPUT'] = Dense(nstates, doutput)

		self.daddress = daddress
		self.dmemory = dmemory

		# ---x--- OLD ----x----
		#self.layers['INPUT'] = Dense(dinput, nstates)
		#self.layers['PREVIOUS_READ'] = Dense(dmemory, nstates)
		
		#self.layers['CONTROL_KEY'] = LSTM(nstates, nstates)
		
		#self.layers['HASH'] = Dense(dmemory, daddress)
		
		#self.layers['CONTENT_KEY_R'] = Dense(nstates, dmemory)
		#self.layers['GATE_R'] = Dense(nstates, daddress)
		#self.layers['LOCATION_R'] = Dense(nstates, daddress) 
		
		#self.layers['CONTENT_KEY_W'] = Dense(nstates, dmemory)
		#self.layers['GATE_W'] = Dense(nstates, daddress)
		#self.layers['LOCATION_W'] = Dense(nstates, daddress) 
		#self.layers['ERASE'] = Dense(nstates, dmemory)
		#self.layers['ADD'] = Dense(nstates, dmemory) 
		#self.layers['OUTPUT'] = Dense(nstates, doutput)

	def __call__(self, input, prev_read):
		layer = self.layers 	# alias
		daddress = self.daddress
		dmemory = self.dmemory

		V = layer['INPUT'](input) + layer['PREVIOUS_READ'](prev_read)
		C = layer['CONTROL_KEY'](V)

		indx = 0
		content_r = C[indx:indx+dmemory]
		indx += dmemory
		content_w = C[indx:indx+dmemory]
		indx += dmemory

		loc_r = C[indx:indx+daddress]
		indx += daddress
		loc_w = C[indx:indx+daddress]
		indx += daddress

		g_r = C[indx:indx+daddress]
		indx += daddress
		g_w = C[indx:indx+daddress]
		indx += daddress

		erase = C[indx:indx+dmemory]
		indx += dmemory
		add = C[indx:indx+dmemory]
		indx += dmemory

		address_r = loc_r * g_r + (1 - g_r) * tanh(layer['HASH'](content_r))
		address_w = loc_w * g_w + (1 - g_w) * tanh(layer['HASH'](content_w))

		output = sigmoid(layer['OUTPUT'](C[indx:]))

		# ---x--- OLD ----x----
		#V = layer['INPUT'](input) + layer['PREVIOUS_READ'](prev_read)
		#C = layer['CONTROL_KEY'](V)

		#content_r = tanh(layer['CONTENT_KEY_R'](C)) #dmemory
		#content_w = tanh(layer['CONTENT_KEY_W'](C)) #dmemory

		#loc_r = tanh(layer['LOCATION_R'](C))	#daddress
		#loc_w = tanh(layer['LOCATION_W'](C))	#daddress

		#g_r = ReLU(layer['GATE_R'](C)) #daddress
		#g_w = ReLU(layer['GATE_W'](C)) #daddress

		#address_r = loc_r * g_r + (1 - g_r) * tanh(layer['HASH'](content_r))
		#address_w = loc_w * g_w + (1 - g_w) * tanh(layer['HASH'](content_w))

		#erase = ReLU(layer['ERASE'](C)) #dmemory
		#add = tanh(layer['ADD'](C))	#dmemory

		#output = sigmoid(layer['OUTPUT'](C))

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
