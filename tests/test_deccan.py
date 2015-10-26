import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

from layers import gaussian, deccan

gaussian_grad = grad(gaussian)
deccan_grad = grad(deccan)

address = 0.0
locations = np.linspace(-1, 1, 1000).reshape(-1, 1)
plt.ion()
plt.figure(1)

triggers = []
for i in range(1000):
	G = gaussian(locations[i].reshape(1, -1), address)
	triggers.append(G.sum())
plt.plot(locations, triggers, label='Gaussian')

print 'Gaussian:'
print '\tmax: ', gaussian(np.array([[0.0]]), address)
print '\tvalue at 2: ', gaussian(np.array([[2.0]]), address)
print '\tnear: ', gaussian_grad(np.array([[1.0]]), address)
print '\tfar: ', gaussian_grad(np.array([[2.0]]), address)


triggers = []
for i in range(1000):
	G = deccan(locations[i].reshape(1, -1), address)
	triggers.append(G.sum())
plt.plot(locations, triggers, label='Deccan')

print 'Deccan: '
print '\tmax: ', deccan(np.array([[0.0]]), address)
print '\tvalue at 2: ', deccan(np.array([[2.0]]), address)
print '\tnear: ', deccan_grad(np.array([[1.0]]), address)
print '\tfar: ', deccan_grad(np.array([[2.0]]), address)

plt.legend()

plt.ion()
plt.figure(2)

triggers = []
for i in range(1000):
	G = gaussian_grad(locations[i].reshape(1, -1), address)
	triggers.append(G.sum())
plt.plot(locations, triggers, label='Gaussian')

triggers = []
for i in range(1000):
	G = deccan_grad(locations[i].reshape(1, -1), address)
	triggers.append(G.sum())
plt.plot(locations, triggers, label='Deccan')

plt.legend()

raw_input()

