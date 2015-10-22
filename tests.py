from sm import *
from autograd import grad

#activations = RBF(np.random.random((1, 4)), np.zeros((20, 4)))

#assert activations.shape == (20, 1), 'RBF Error'

Ganesha = SpatialMemoryMachine(1, 4, 2)
print Ganesha.loss(np.random.randn(5, 1), np.random.randn(5, 1))

g = grad(Ganesha.loss)

print g(np.random.randn(5, 1), np.random.randn(5, 1))