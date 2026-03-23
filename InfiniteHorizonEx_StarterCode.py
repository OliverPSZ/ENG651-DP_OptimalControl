import numpy as np
import matplotlib.pyplot as plt

# Problem data
k = 20  # Grid size
m = 2

X = np.zeros((k, k))
X[4, 4] = 1
X[16, 9] = 1
X[9, 14] = 1  # Target states

G = np.zeros((k, k, m))  # Stage cost
G[:, :, 0] = 1  # Holding cost
G[4, 4, 1] = -120
G[16, 9, 1] = -70
G[9, 14, 1] = -150  # Stopping cost

# DP algorithm
J = np.zeros((k, k))  # Optimal cost functions
pistar = np.zeros((k, k), dtype=int)  # Optimal policy

Jplus = np.random.rand(k, k)  # Placeholder for optimal cost update

#####################################
# YOUR INFINITE HORIZON DP CODE HERE
#####################################

# Plot optimal policy and value function
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(pistar, cmap='hot', extent=[0, k - 1, 0, k - 1], origin='lower')
plt.title('Optimal Policy')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(J, cmap='copper', extent=[0, k - 1, 0, k - 1], origin='lower')
plt.title('Optimal Value Function')
plt.colorbar()
plt.show()
