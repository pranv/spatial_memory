import autograd.numpy as np


def sigmoid(X):
	return 1.0 / (1.0 + np.exp(-X))


def orthogonalize(n):
    W = np.random.randn(n, n)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')


def softmax(X):
	n = np.exp(X)
	return n / np.sum(n)


def softplus(X):
	return np.log(1 + np.exp(X))


class LSTM(object):
	def __init__(self, dinput, nstates, fbias=0):
		self.dinput = dinput
		self.nstates = nstates

		W = np.random.random((dinput + nstates + 1, nstates * 4)) / np.sqrt(dinput + nstates)
		W[dinput:-1, 0 * nstates : 1 * nstates] = orthogonalize(nstates)
		W[dinput:-1, 1 * nstates : 2 * nstates] = orthogonalize(nstates)
		W[dinput:-1, 2 * nstates : 3 * nstates] = orthogonalize(nstates)
		W[dinput:-1, 3 * nstates : 4 * nstates] = orthogonalize(nstates)
		W[-1, :] = 0 
		W[-1, 2 * nstates : 3 * nstates] = fbias
		self.W = W

		self.prev_c = np.zeros(nstates)
		self.prev_Y = np.zeros(nstates)
	
	def __call__(self, X):
		V = np.concatenate([X, self.prev_Y, np.ones(1)])
		S = np.dot(V, self.W) 

		z, i, f, o = np.split(S, 4)
		Z, I, F, O = sigmoid(z), np.tanh(i), sigmoid(f), sigmoid(o)
		
		c = Z * I + F * self.prev_c
		
		C = np.tanh(c)
		Y = O * C

		self.prev_c = c
		self.prev_Y = Y

		return Y

	def get_prev_c(self):
		return self.prev_c

	def get_prev_Y(self):
		return self.prev_Y

	def get_params(self):
		return self.W

	def set_prev_c(self, prev_c):
		self.prev_c = prev_c

	def set_prev_Y(self, prev_Y):
		self.prev_Y = prev_Y

	def set_params(self, params):
		self.W = params


class Dense(object):
	def __init__(self, dinput, doutput):
		self.dinput = dinput
		self.doutput = doutput

		W = np.random.random((dinput + 1, doutput)) 
		W /= np.sqrt(dinput + doutput)
		self.W = W

	
	def __call__(self, X):
		V = np.concatenate([X, np.ones(1)])
		return np.dot(V, self.W)

	def get_params(self):
		return self.W

	def set_params(self, params):
		self.W = params