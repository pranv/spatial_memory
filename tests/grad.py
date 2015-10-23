import autograd.numpy as np
from autograd import grad

from smm import SpatialMemoryMachine
import generator
from RMSProp import RMSProp


def lossfun(params):
	machine.set_params(params)
	loss = 0
	for t in range(inputs.shape[0]):
		input = inputs[t]
		target = targets[t]
		output = machine.forward(input)
        loss -= np.sum(target * np.log2(output) + (1 - target) * np.log2(1 - output))
	
	return loss 


def gradCheck(params, deltas, inputs, targets, epsilon, tolerance):
  """
  Finite difference based gradient checking.
  """

  diffs = getDiffs(params, deltas, inputs, targets, epsilon, tolerance)
  answer = True

  for diffTensor, name, delta in zip(diffs, params, deltas):

    if np.abs(diffTensor.max()) >= tolerance:
      print "DIFF CHECK FAILS FOR TENSOR: ", name
      print "DIFF TENSOR: "
      print diffTensor
      print "NUMERICAL GRADIENTS: "
      # diff = grad - delta => diff+delta = grad
      print diffTensor + delta
      print "BPROP GRADIENTS: "
      print delta
      answer = False
    else:
      pass

  return answer

def getDiffs(params, deltas, inputs, targets, epsilon, tolerance):
	"""
	For every (weight,delta) combo in zip(weights, deltas):
	Add epsilon to that weight and compute the loss (first_loss)
	Remove epsilon from that weight and compute the loss (second_loss)
	Check how close (first loss - second loss) / 2h is to the delta from bprop
	"""

	diff_tensors = []
	for d in deltas:
		D = deltas[d]
		diff_tensors.append(np.zeros_like(D))

	for w,d,diffs in zip(params, deltas, diff_tensors):
		W = params[w]
		D = deltas[d]
		print w
		if len(W.shape) == 2:
			for i in range(W.shape[0]):
				for j in range(W.shape[1]):
					W[i,j] += epsilon
					loss  = lossfun(params)
					loss_plus = np.sum(loss)

					W[i,j] -= epsilon*2
					loss = lossfun(params)
					loss_minus = np.sum(loss)

					grad = (loss_plus - loss_minus) / (2 * epsilon)
					diffs[i,j] = grad - D[i,j]

					if np.abs(diffs[i, j]) >= tolerance:
						print 'Death'

					W[i,j] += epsilon

		if len(W.shape) == 1:
			for i in range(W.shape[0]):
				W[i] += epsilon
				loss  = lossfun(params)
				loss_plus = np.sum(loss)

				W[i] -= epsilon*2
				loss = lossfun(params)
				loss_minus = np.sum(loss)

				grad = (loss_plus - loss_minus) / (2 * epsilon)
				diffs[i] = grad - D[i]

				if np.abs(diffs[i]) >= tolerance:
					print 'Death'

				W[i] += epsilon

	return diff_tensors



task = 'copy'
vector_size = 10
seqence_length_min = 15
seqence_length_max = 25

dmemory = vector_size
daddress = 10
dinput = vector_size + 2
doutput = vector_size
nstates = 100
write_threshold = 1e-40

lr = 1e-6
niter = 10000


data = generator.Generator(task, vector_size, seqence_length_min, seqence_length_max)
machine = SpatialMemoryMachine(dmemory, daddress, nstates, dinput, doutput, write_threshold)

inputs, targets, T = data.make()

dW = grad(lossfun)
W = machine.get_params()

gradCheck(W, dW(W), inputs, targets, 10e-4, 10e-4)
