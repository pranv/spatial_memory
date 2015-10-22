task = 'copy'
vector_size = 2
seqence_length_min = 5
seqence_length_max = 10

dmemory = vector_size
daddress = 1
dinput = vector_size + 2
doutput = vector_size
nstates = 100
write_threshold = 1e-20

lr = 4e-4
niter = 2000
decay = 0.9

import autograd.numpy as np
from autograd import grad

from smm import SpatialMemoryMachine
import generator
from RMSProp import RMSProp


data = generator.Generator(task, vector_size, seqence_length_min, seqence_length_max)
machine = SpatialMemoryMachine(dmemory, daddress, nstates, dinput, doutput, write_threshold)

def test(params):
	machine.clear()
	machine.set_params(params)
	inputs, targets, T = data.make()
	loss = 0
	for t in range(inputs.shape[0]):
		input = inputs[t]
		target = targets[t]
		output = machine.forward(input)
		loss -= np.sum(target * np.log2(output) + (1 - target) * np.log2(1 - output))

	loss = loss / ((T * 2 + 2) * vector_size)
	return loss 

def loss(params):
	machine.clear()
	machine.set_params(params)
	inputs, targets, T = data.make()
	loss = 0
	for t in range(inputs.shape[0]):
		input = inputs[t]
		target = targets[t]
		output = machine.forward(input)
		loss -= np.sum(target * np.log2(output) + (1 - target) * np.log2(1 - output))
	
	return loss 


dW = grad(loss)
W = machine.get_params()
optimizer = RMSProp(W, learning_rate=lr, decay=decay, blend=0.95)

for i in range(niter):
	grads = dW(W)
	optimizer(W, grads)
	L = test(W) 
	G = [(grads[g] * grads[g]).sum() / np.prod(grads[g].shape) for g in grads]
	print '\t|', i, '\t|  loss: ', L, '\t|  gradient norm: ', sum(G) / len(G), '\t|'


machine.set_params(W)
machine.clear()
inputs, targets, T = data.make()
outputs = np.zeros_like(targets)
for t in range(inputs.shape[0]):
	input = inputs[t]
	target = targets[t]
	output = machine.forward(input)
	outputs[t] += output
	print '----' * 20
	print input
	print output
	print target

print inputs
print outputs
print targets
print T