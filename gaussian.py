import numpy as np

import matplotlib.pyplot as plt
plt.ion()

def activate(locations, address):
	# Gaussian
	sigma = 0.01
	mu = address
	norm = (locations - mu) ** 2
	fac = -np.sum(norm / (2 * (sigma ** 2)), axis=1, keepdims=True)
	return np.exp(fac)


address = 0.0
locations = np.linspace(-0.2, 0.2, 10000).reshape(-1, 1)
triggers = activate(locations, address)
plt.plot(locations, triggers)

print 'max: ', activate(np.array([[address]]), address)

address = 0.1
locations = np.linspace(-0.2, 0.2, 10000).reshape(-1, 1)
triggers = activate(locations, address)
plt.plot(locations, triggers)

print 'max: ', activate(np.array([[address]]), address)

raw_input()