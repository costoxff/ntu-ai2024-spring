import numpy as np

dim1 = (1,)
print(*dim1)
s1 = np.zeros((5, *dim1))
print(s1.shape)

dim2 = (2, 2)
print(*dim2)
s2 = np.zeros((5, *dim2))
print(s2.shape)

# print(s2)

ind = np.random.randint(0, 10, 5)
print(ind)
sel = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(sel)
print(sel[ind])