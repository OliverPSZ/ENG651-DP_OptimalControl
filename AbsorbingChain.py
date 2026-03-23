import numpy as np

# Problem data
p = np.array([0.4, 0.3, 0.3])
P = np.array([
    [*p, 0, 0, 0, 0],
    [0, *p, 0, 0, 0],
    [0, 0, *p, 0, 0],
    [0, 0, 0, *p, 0],
    [0, 0, 0, 0, *p],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1]
])
P11 = P[:5, :5]
P12 = P[:5, 5:7]
P22 = P[5:7, 5:7]

# Limit matrix
L = np.linalg.inv(np.eye(5) - P11).dot(P12)
print(L)
