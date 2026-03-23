import numpy as np

# Problem data
n = 5
m = 2
T = 30

x0 = 40 * np.random.randn(n, 1)

A = np.random.randn(n, n)
A = A / (max(abs(np.linalg.eig(A)[0])))
B = np.random.randn(n, m)

Q = 1 * np.eye(n)
QT = 5 * np.eye(n)
R = 1 * np.eye(m)

W = 0.1 * np.eye(n)

# Optimal closed-loop feedback policy
P = np.zeros((n, n, T + 1))
r = np.zeros(T + 1)
K = np.zeros((m, n, T))

# Dynamic programming (DP) algorithm
# YOUR DP CODE HERE


# plotting
# plot some interesting quantities!
# e.g., plot cost coefficient, gain coefficients, a sample trajectory,
# compare with open-loop optimal control sequence, Monte Carlo optimal cost estimate, etc.
