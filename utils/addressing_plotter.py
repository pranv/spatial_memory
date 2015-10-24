import matplotlib.pyplot as plt
plt.ion()

A = []


B = []
for b in range(1,len(A) + 1, 2):
    B.append(A[b])

C = []
for b in range(0,len(A), 2):
    C.append(A[b])

plt.clf()
plt.plot(B)
plt.plot(C)