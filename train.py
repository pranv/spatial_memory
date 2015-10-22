task = 'copy'
vector_size = 10
seqence_length_min = 5
seqence_length_max = 15

dmemory = vector_size
daddress = 2
dinput = vector_size + 2
doutput = vector_size
nstates = 100
write_threshold = 1e-2


import autograd.numpy as np
from autograd import grad

from smm import SpatialMemoryMachine
import generator
from RMSProp import RMSProp

data = generator.Generator(task, vector_size, seqence_length_min, seqence_length_max)

machine = SpatialMemoryMachine(dmemory, daddress, nstates, dinput, doutput, write_threshold)

def loss(params):
	machine.set_params(params)
	inputs, targets, T = data.make()
	ep = 2**-23
	loss = 0
	for t in range(T):
		input = inputs[t]
		target = targets[t]
		output = machine.forward(input)
        loss += np.sum(target * np.log2(output + ep) + (1 - target) * np.log2(1 - output + ep))
        print loss

	return -loss


dW = grad(loss)
W = machine.get_params()
optimizer = RMSProp(W, learning_rate=10e-5, decay=0.95, blend=0.95)

while True:
	optimizer(W, dW(W))
