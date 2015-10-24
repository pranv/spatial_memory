import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
plt.ion()

def gaussian(locations, address):
	sigma = 0.05
	mu = address
	norm = (locations - mu) ** 2
	fac = -np.sum(norm / (2 * (sigma ** 2)))
	return np.exp(fac)


def deccan(locations, mu):
	sigma = 0.05
	mu = address
	norm = (locations - mu) ** 2
	fac = -np.sum(norm / (2 * (sigma ** 2)))
	return (np.exp(fac) + sigma / (norm + 0.999))


gaussian_grad = grad(gaussian)
deccan_grad = grad(deccan)

address = 0.0
locations = np.linspace(-2, 2, 1000).reshape(-1, 1)

triggers = []
for i in range(1000):
	G = gaussian(locations[i], address)
	triggers.append(G)

plt.plot(locations, triggers, label='Gaussian')

print 'Gaussian:'
print '\tmax: ', gaussian(np.array([[0.0]]), address)
print '\tvalue at .50: ', gaussian(np.array([[-5.0]]), address)
print '\tnear: ', gaussian_grad(np.array([[-1.0]]), address)
print '\tfar: ', gaussian_grad(np.array([[-5.0]]), address)


triggers = []
for i in range(1000):
	G = deccan(locations[i], address)
	triggers.append(G)

plt.plot(locations, triggers, label='Deccan')

print 'Deccan: '
print '\tmax: ', deccan(np.array([[0.0]]), address)
print '\tvalue at .50: ', deccan(np.array([[-0.50]]), address)
print '\tnear: ', deccan_grad(np.array([[-1.0]]), address)
print '\tfar: ', deccan_grad(np.array([[-0.50]]), address)

plt.legend()

raw_input()