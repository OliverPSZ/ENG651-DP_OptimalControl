# This script demonstrates the implementation of a basic dynamic programming algorithm
# to solve a finite horizon MDP with randomly generated transition probabilities and cost functions.
# The optimal policy and cost functions are computed through backward recursion.

import numpy as np

# Problem data
n = 100
m = 20
T = 50

P = np.random.rand(n, n, m)  # Transition matrices

# Normalize to get row stochastic matrices
for i in range(m):
    P[:, :, i] = np.diag(1.0 / np.sum(P[:, :, i], axis=1)) @ P[:, :, i]

G = np.random.rand(n, m, T)  # Stage cost
GT = np.random.rand(n)  # Terminal cost

# DP algorithm
J = np.zeros((n, T + 1))  # Optimal cost functions
pistar = np.zeros((n, T), dtype=int)  # Optimal policy

# Initialize
J[:, T] = GT

# Recursion
for t in range(T - 1, -1, -1):  # Backward time recursion
    for i in range(n):  # State loop
        Ju = np.zeros(m)
        for j in range(m):  # Input loop
            Ju[j] = G[i, j, t] + np.dot(P[i, :, j], J[:, t + 1])
        J[i, t] = np.min(Ju)
        pistar[i, t] = np.argmin(Ju)

print("J:", J)
print("pistar:", pistar)
