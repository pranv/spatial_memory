import autograd.numpy as np

def sigmoid(X):
	return 1.0 / (1.0 + np.exp(-X))


def tanh(X):
	return np.tanh(X)


def ReLU(X):
    Y = X + np.sqrt(X * X)
    return Y / 2

def softmax(X):
	n = np.exp(X)
	return n / np.sum(n)


def softplus(X):
	return np.log(1 + np.exp(X))


def gaussian(X, Y, sigma=0.01):
	norm = np.sum((X - Y) ** 2, axis=1, keepdims=True)
	return np.exp(-norm / (2 * (sigma ** 2)))


def deccan(X, Y, sigma=0.01):
	norm = np.sum((X - Y) ** 2, axis=1, keepdims=True)
	return (np.exp(-norm / (2 * (sigma ** 2))))


def orthogonalize(W):
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')


class Dense(object):
	def __init__(self, dinput, doutput):
		self.dinput = dinput
		self.doutput = doutput
		
		W = np.random.random((dinput + 1, doutput)) / np.sqrt(dinput + doutput)
		self.W = W
		
		print 'Linear layer with ', np.prod(W.shape), 'parameters'

	def __call__(self, X):
		V = np.concatenate([X, np.ones(1)])
		return np.dot(V, self.W)

	def get_params(self):
		return self.W.flatten()

	def set_params(self, params):
		self.W = params.reshape(self.W.shape)

	def clear(self):
		pass


class LSTM(object):
	def __init__(self, dinput, nstates, fbias=1.0):
		self.dinput = dinput
		self.nstates = nstates

		W = np.random.random((dinput + nstates + 1, nstates * 4)) / np.sqrt(dinput + nstates)
		W[dinput:-1, 0 * nstates : 1 * nstates] = orthogonalize(W[dinput:-1, 0 * nstates : 1 * nstates])
		W[dinput:-1, 1 * nstates : 2 * nstates] = orthogonalize(W[dinput:-1, 1 * nstates : 2 * nstates])
		W[dinput:-1, 2 * nstates : 3 * nstates] = orthogonalize(W[dinput:-1, 2 * nstates : 3 * nstates])
		W[dinput:-1, 3 * nstates : 4 * nstates] = orthogonalize(W[dinput:-1, 3 * nstates : 4 * nstates])
		W[-1, :] = 0 
		W[-1, 2 * nstates : 3 * nstates] = fbias
		self.W = W

		self.c0 = np.random.random((nstates))
		self.Y0 = np.random.random((nstates))

		self.c, self.Y = self.c0, self.Y0 

		print 'LSTM layer with ', np.prod(W.shape), 'parameters'
	
	def __call__(self, X):
		V = np.concatenate([X, self.Y, np.ones(1)])
		S = np.dot(V, self.W) 

		z, i, f, o = np.split(S, 4)
		Z, I, F, O = sigmoid(z), np.tanh(i), sigmoid(f), sigmoid(o)
		
		self.c = Z * I + F * self.c
		
		C = np.tanh(self.c)
		self.Y = O * C

		return self.Y

	def get_params(self):
		w, c, y = self.W.flatten(), self.c0.flatten(), self.Y0.flatten()
		return np.concatenate([w, c, y], axis=1)

	def set_params(self, params):
		nw, nc = np.prod(self.W.shape), np.prod(self.c0.shape)
		W, c0, Y0 = np.split(params, [nw, nw+nc])
		self.W = W.reshape(self.W.shape)
		self.c0 = c0.reshape(self.c0.shape)
		self.Y0 = Y0.reshape(self.Y0.shape)

	def clear(self):
		self.c = self.c0
		self.Y = self.Y0