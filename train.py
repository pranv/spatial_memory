import autograd.numpy as np
from autograd import grad

from machine import SpatialMemoryMachine
from utils import generator
from climin import RmsProp, Adam

# all hyper parameters
task = 'copy'
vector_size = 1
seqence_length_min = 1
seqence_length_max = 5

dmemory = vector_size
daddress = 1
nstates = 100
dinput = vector_size + 2
doutput = vector_size
init_units = 5
create_memories = False
influence_threshold = 0.01
sigma = 0.1

lr = 1e-4
decay = 0.5
momentum = 0.7
grad_clip = (-10, 10)

niter = 10000
batch_size = 1	# not actual batches. manual summing of gradients

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

print loss(W), L

opt = Adam(W, dW)#, lr, momentum=momentum, decay=decay)

for i in opt:
	print i['n_iter'],  i['n_iter'] * batch_size, 'loss: ', L[0].value
	
	if i['n_iter'] > niter:
		break
