task = 'copy'
vector_size = 10
seqence_length_min = 15
seqence_length_max = 25

dmemory = vector_size
daddress = 1
dinput = vector_size + 2
doutput = vector_size
nstates = 100
write_threshold = 0.5
sigma = 0.05

lr = 1e-4
niter = 2000
decay = 0.9
blend = 0.95

ep = 2e-23

import autograd.numpy as np
from autograd import grad
from smm import SpatialMemoryMachine
import generator
from RMSProp import RMSProp

data = generator.Generator(task, vector_size, seqence_length_min, seqence_length_max)
machine = SpatialMemoryMachine(dmemory, daddress, nstates, dinput, doutput, write_threshold, sigma)

def test(params):
	machine.clear()
	machine.set_params(params)
	inputs, targets, T = data.make()
	loss = 0
	for t in range(inputs.shape[0]):
		input = inputs[t]
		target = targets[t]
		output = machine.forward(input, True)
		loss -= np.sum(target * np.log2(output + ep) + (1 - target) * np.log2(1 - output + ep))

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
		loss -= np.sum(target * np.log2(output + ep) + (1 - target) * np.log2(1 - output + ep))
	
	return loss 


dW = grad(loss)
W = machine.get_params()
optimizer = RMSProp(W, learning_rate=lr, decay=decay, blend=blend)

for i in range(niter):
	grads = dW(W)
	optimizer(W, grads)
	L = test(W) 
	G = [(grads[g] * grads[g]).sum() / np.prod(grads[g].shape) for g in grads]
	print '\t', i, '\tloss: ', L, '\tgradient norm: ', sum(G) / len(G), '\t'


machine.set_params(W)
machine.clear()
inputs, targets, T = data.make()
outputs = np.zeros_like(targets)
for t in range(inputs.shape[0]):
	input = inputs[t]
	target = targets[t]
	output = machine.forward(input, True)
	outputs[t] += output
	print '----' * 20
	print input
	print output
	print target