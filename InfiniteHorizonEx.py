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
t = 0  # Value iteration counter

# Recursion
while np.linalg.norm(Jplus - J) > 1e-8:  # Value iteration
    t += 1
    J = Jplus.copy()
    for i in range(k):
        for j in range(k):
            # Interior states
            if (0 < i < k - 1) and (0 < j < k - 1):
                Ju = np.zeros(m)
                Ju[0] = G[i, j, 0] + (1 / 4) * np.sum([J[i + 1, j], J[i - 1, j], J[i, j + 1], J[i, j - 1]])
                Ju[1] = G[i, j, 1]
                Jplus[i, j] = np.min(Ju)
                pistar[i, j] = np.argmin(Ju)

            # Boundary states
            elif (i == 0) and (0 < j < k - 1):
                Ju = np.zeros(m)
                Ju[0] = G[i, j, 0] + (1 / 3) * np.sum([J[i + 1, j], J[i, j + 1], J[i, j - 1]])
                Ju[1] = G[i, j, 1]
                Jplus[i, j] = np.min(Ju)
                pistar[i, j] = np.argmin(Ju)

            elif (i == k - 1) and (0 < j < k - 1):
                Ju = np.zeros(m)
                Ju[0] = G[i, j, 0] + (1 / 3) * np.sum([J[i - 1, j], J[i, j + 1], J[i, j - 1]])
                Ju[1] = G[i, j, 1]
                Jplus[i, j] = np.min(Ju)
                pistar[i, j] = np.argmin(Ju)

            elif (j == 0) and (0 < i < k - 1):
                Ju = np.zeros(m)
                Ju[0] = G[i, j, 0] + (1 / 3) * np.sum([J[i + 1, j], J[i - 1, j], J[i, j + 1]])
                Ju[1] = G[i, j, 1]
                Jplus[i, j] = np.min(Ju)
                pistar[i, j] = np.argmin(Ju)

            elif (j == k - 1) and (0 < i < k - 1):
                Ju = np.zeros(m)
                Ju[0] = G[i, j, 0] + (1 / 3) * np.sum([J[i + 1, j], J[i - 1, j], J[i, j - 1]])
                Ju[1] = G[i, j, 1]
                Jplus[i, j] = np.min(Ju)
                pistar[i, j] = np.argmin(Ju)

            # Corner states
            elif (i == 0) and (j == 0):
                Ju = np.zeros(m)
                Ju[0] = G[i, j, 0] + (1 / 2) * np.sum([J[i + 1, j], J[i, j + 1]])
                Ju[1] = G[i, j, 1]
                Jplus[i, j] = np.min(Ju)
                pistar[i, j] = np.argmin(Ju)

            elif (i == 0) and (j == k - 1):
                Ju = np.zeros(m)
                Ju[0] = G[i, j, 0] + (1 / 2) * np.sum([J[i + 1, j], J[i, j - 1]])
                Ju[1] = G[i, j, 1]
                Jplus[i, j] = np.min(Ju)
                pistar[i, j] = np.argmin(Ju)

            elif (i == k - 1) and (j == 0):
                Ju = np.zeros(m)
                Ju[0] = G[i, j, 0] + (1 / 2) * np.sum([J[i - 1, j], J[i, j + 1]])
                Ju[1] = G[i, j, 1]
                Jplus[i, j] = np.min(Ju)
                pistar[i, j] = np.argmin(Ju)

            elif (i == k - 1) and (j == k - 1):
                Ju = np.zeros(m)
                Ju[0] = G[i, j, 0] + (1 / 2) * np.sum([J[i - 1, j], J[i, j - 1]])
                Ju[1] = G[i, j, 1]
                Jplus[i, j] = np.min(Ju)
                pistar[i, j] = np.argmin(Ju)

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
