import numpy as np

class RMSProp(object):
	def __init__(self, W, learning_rate=1e-4, alpha=0.95, momentum=0.9, epsilon=1e-8, grad_clip=(-10, 10)):
		self.lr = learning_rate
		self.alpha = alpha
		self.momentum = momentum
		self.epsilon = epsilon
		self.grad_clip = grad_clip
		
		self.n  = {}
		self.g  = {}
		self.delta = {}
		for key, value in W.iteritems():
			self.n[key]  = np.zeros_like(value)
			self.g[key]  = np.zeros_like(value)
			self.delta[key] = np.zeros_like(value)

	def __call__(self, W, dW):
		n, g, delta = self.n, self.g, self.delta
		for k in W:
			dW[k] = np.clip(dW[k], *self.grad_clip)
			n[k] *= self.alpha
			n[k] += (1 - self.alpha) * (dW[k] ** 2)
			g[k] *= self.alpha
			g[k] += (1 - self.alpha) * dW[k]
			delta[k] *= self.momentum
			delta[k] -= self.lr * dW[k] / np.sqrt(n[k] - g[k] * g[k] + self.epsilon)
			W[k] += delta[k]