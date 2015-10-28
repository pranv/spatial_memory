import autograd.numpy as np
from autograd import grad

from machine import SpatialMemoryMachine
from utils import generator

from climin import RmsProp

# all hyper parameters
task = 'copy'
vector_size = 4
seqence_length_min = 10
seqence_length_max = 20

dmemory = vector_size
daddress = 1
nstates = 5
dinput = vector_size + 2
doutput = vector_size
init_units = 20
create_memories = False
influence_threshold = 0.1
sigma = 0.01

lr = 1e-3
decay = 0.9
momentum = 0.9
grad_clip = (-10, 10)

niter = 10000
batch_size = 20	# not actual batches. manual summing of gradients

data = generator.Generator(task, vector_size, seqence_length_min, seqence_length_max)
M = SpatialMemoryMachine(dmemory, daddress, nstates, dinput, doutput, init_units, create_memories, influence_threshold, sigma)

L = [1.4]

def loss(W):
	M.set_params(W)
	l, T = 0, 0
	for j in range(batch_size):
		M.clear()
		inputs, targets, t = data.make()
		l += M.loss(inputs, targets)
		T += t
	L[0] = l / ((T * 2 + 2) * vector_size)
	return l 

W = M.get_params()
dW = grad(loss)

opt = RmsProp(W, dW, lr, momentum=momentum, decay=decay)

for i in opt:
	print i['n_iter'],  i['n_iter'] * batch_size, 'loss: ', L[0].value
	
	if i['n_iter'] > niter:
		break
