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

		self.forget()
	
	def __call__(self, X):
		V = np.concatenate([X.reshape(1, -1), self.prev_Y, np.ones((1, 1))], axis=1)
		S = np.dot(V, self.W) 
		
		z, i, f, o = np.split(S, 4, axis=1)
		Z, I, F O = sigmoid(z), np.tanh(i), sigmoid(f), sigmoid(o)
		
		c = Z * I + F * self.prev_c
		
		C = np.tanh(c)
		Y = O * C

		self.prev_c = c
		self.prev_Y = Y

		return Y

	def forget(self):
		self.prev_c = np.zeros((1, nstates))
		self.prev_Y = np.zeros((1, nstates))


class Dense(object):
	def __init__(self, dinput, doutput):
		self.dinput = dinput
		self.doutput = doutput

		W = np.random.random((dinput + doutput + 1, doutput)) / np.sqrt(dinput + doutput)
		self.W = W

	
	def __call__(self, X):
		return np.dot(X, self.W)