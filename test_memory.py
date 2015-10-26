import numpy as np
import generator
import matplotlib.pyplot as plt

from memory import Memory

def memIO_test():
	task = 'copy'
	vector_size = 1
	seqence_length_min = 4
	seqence_length_max = 8

	dmemory = vector_size
	daddress = 1
	dinput = vector_size + 2
	doutput = vector_size
	nstates = 50
	influence_threshold = 0.1
	init_units = seqence_length_max
	sigma = 0.01

	lr = 1e-4

	memory = Memory(dmemory, daddress, init_units, False, influence_threshold, sigma)
	data = generator.Generator(task, vector_size, seqence_length_min, seqence_length_max)

	inputs, targets, T = data.make()
	outputs = np.ones_like(targets) * 0.0

	for t in range(1, T + 1):
		memory.commit(memory.locations[t -1], 1, inputs[t][:-2])
		outputs[t] = 0.0

	print '----' * 20

	for t in range(T + 2, (2 * T + 2)):
		outputs[t] = a = memory.fetch(memory.locations[(t - T - 2)])
		
	for t in range(inputs.shape[0]):
		input = inputs[t]
		target = targets[t]
		output = outputs[t]
		if np.abs(target - output) > influence_threshold:
			print inputs
			print outputs
			print targets
			print memory.values
			print memory.locations
			raise ValueError
			print '----' * 20

	print 'Read Write Operations are clear'

def locations_test():
	MEM = Memory(dmemory=4, daddress=1, init_units=25, create_memories=True, influence_threshold=0.1, sigma=0.01)
	address = np.array([[0.2]])
	
	memory = np.ones((1, 4))
	MEM.commit(address, erase=0, add=memory)
	print MEM.values, MEM.locations
	
	MEM.commit(address, erase=0, add=memory)
	print MEM.values, MEM.locations

	MEM.commit(address + 0.001, erase=0, add=memory)
	print MEM.values, MEM.locations

	print 'memory activation: ', MEM.activate(address)

	MEM.commit(address + 0.5, erase=0, add=memory)
	print MEM.values, MEM.locations

	locations = np.linspace(-2, 2, 1000)
	activations = np.array([MEM.activate(locations[i]) for i in range(1000)])
	print activations.shape
	plt.ion()
	plt.plot(locations, activations.sum(axis=1))

	raw_input()

memIO_test()
locations_test()
