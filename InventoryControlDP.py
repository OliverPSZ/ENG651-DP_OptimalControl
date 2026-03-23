import numpy as np
import matplotlib.pyplot as plt

# Problem data
C = 6  # Warehouse capacity
p = np.array([0.1, 0.2, 0.7])  # Demand probability distribution

n = C + 1  # Number of states
m = n
T = 50  # Time horizon

# Cost parameters
s = 0.1  # Unit stock storage cost
o = 1  # Fixed ordering cost

g = np.hstack((s * np.arange(C + 1).reshape(C+1, 1), np.vstack((np.ones(C), np.tile(s * np.arange(C + 1) + o, (6,1)))))  # Stage cost
print(g.shape)
gT = np.zeros(n)

# DP algorithm
J = np.zeros((n, T + 1))  # Optimal cost functions
pistar = np.zeros((n, T), dtype=int)  # Optimal policy

# Initialize
J[:, T] = gT

# Recursion
for t in range(T - 1, -1, -1):  # Backward time recursion
    for i in range(n):  # State loop
        Ju = np.full(m, np.inf)
        for j in range(max(3 - i, 0), min(n - i, n)):  # Input loop, only update for allowable inputs
            pu = np.zeros(n)
            pu[j + i - 2:j + i + 1] = p
            Ju[j] = g[i * (C + 1) + j] + np.dot(pu, J[:, t + 1])
        J[i, t] = np.min(Ju)
        pistar[i, t] = np.argmin(Ju)

# Optimal cost
plt.plot(np.arange(C + 1), J[:, 0], linewidth=3)
plt.xlabel('Inventory level')
plt.ylabel('Optimal cost')
plt.xticks(np.arange(C + 1))
plt.grid(True)
plt.show()

# Optimal policy
plt.bar(np.arange(C + 1), pistar[:, 0])
plt.xlabel('Inventory level')
plt.ylabel('Optimal order amount')
plt.xticks(np.arange(C + 1))
plt.grid(True)
plt.show()
