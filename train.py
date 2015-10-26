import autograd.numpy as np
import generator
from autograd import grad
from machine import SpatialMemoryMachine
from RMSProp import RMSProp

import matplotlib.pyplot as plt

# all hyper parameters
task = 'copy'
vector_size = 2
seqence_length_min = 10
seqence_length_max = 20

dmemory = vector_size
daddress = 1
nstates = 20
dinput = vector_size + 2
doutput = vector_size
init_units = 25
create_memories = False
influence_threshold = 0.1
sigma = 0.01

lr = 1e-4
alpha = 0.95
momentum = 0.9
grad_clip = (-1, 1)

niter = 1000
batch_size = 100
print_every = 2

data = generator.Generator(task, vector_size, seqence_length_min, seqence_length_max)
M = SpatialMemoryMachine(dmemory, daddress, nstates, dinput, doutput, init_units, create_memories, influence_threshold, sigma)

def loss(W):
	M.set_params(W)
	M.clear()
	inputs, targets, T = data.make()
	loss = M.loss(inputs, targets)
	return loss 

W = M.get_params()
dW = grad(loss)
optimizer = RMSProp(W, learning_rate=lr, alpha=alpha, momentum=momentum, epsilon=1e-8, grad_clip=grad_clip)

# the training loop
plt.ion()
losses = [1.4]
plt.plot(losses)
plt.draw()
for i in range(niter):
	grads = dW(W)
	for j in range(batch_size - 1):
		_grads = dW(W)
		for g in grads:
			grads[g] += _grads[g]

	for g in grads:
			grads[g] *= (1.0 / batch_size)
	
	optimizer(W, grads)	

	if i % print_every == 0:
		L = 0
		for j in range(batch_size):
			L += loss(W) / ((seqence_length_max * 2 + 2) * vector_size)
		L = L / batch_size
		print 'loss at iteration ', i * batch_size * print_every, ': ', L
		
		M.clear()
		M.set_params(W)
		inputs, targets, T = data.make()
		outputs = M(inputs)
		print inputs
		print outputs
		print targets
		
		losses.append(L)
		plt.clf()
		plt.plot(losses)
		plt.draw()

		for g in grads:
			G = (grads[g] * grads[g]).sum() / np.prod(grads[g].shape)
			print '\tgradient norm for ', g, ' : ', G
