import autograd.numpy as np
import generator
from autograd import grad
from machine import SpatialMemoryMachine
from RMSProp import RMSProp


# all hyper parameters
task = 'copy'
vector_size = 1
seqence_length_min = 4
seqence_length_max = 8

dmemory = vector_size
daddress = 1
nstates = 100
dinput = vector_size + 2
doutput = vector_size
init_units = 10
create_memories = False
influence_threshold = 0.1
sigma = 0.01

lr = 1e-4
alpha = 0.95
momentum = 0.9
grad_clip = (-10, 10)

niter = 100
batch_size = 10
print_every = 10

data = generator.Generator(task, vector_size, seqence_length_min, seqence_length_max)
M = SpatialMemoryMachine(dmemory, daddress, nstates, dinput, doutput, init_units, create_memories, influence_threshold, sigma)

def loss(W):
	M.clear()
	M.set_params(W)
	inputs, targets, T = data.make()
	loss = M.loss(inputs, targets)
	return loss 

W = M.get_params()
dW = grad(loss)
optimizer = RMSProp(W, learning_rate=lr, alpha=alpha, momentum=momentum, epsilon=1e-8, grad_clip=grad_clip)

# the training loop
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
		L = loss(W) / ((seqence_length_max * 2 + 2) * vector_size)
		G = [(grads[g] * grads[g]).sum() / np.prod(grads[g].shape) for g in grads]
		print '\t', i * batch_size, '\tloss: ', L, '\tgradient norm: ', sum(G) / len(G), '\t'
