import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

# Problem data
n = 15
m = 4

A = np.random.randn(n, n)
B = np.random.randn(n, m)

# Check for controllability
is_controllable = np.linalg.matrix_rank(np.hstack((B, A @ B, A @ A @ B, A @ A @ A @ B))) == n
print("Controllability Check:", is_controllable)

Q = np.eye(n)
R = np.eye(m)

W = np.eye(n)

# Optimal closed-loop feedback policy using value iteration
P = np.zeros((n, n))
Pplus = np.random.randn(n, n)

# Value iteration recursion
t = 0
while np.linalg.norm(P - Pplus, 'fro') > 1e-6:
    t += 1
    P = Pplus
    Pplus = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

K = -np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A


# compare with built-in scipy function
P1 = solve_discrete_are(A, B, Q, R)
K1 = -np.linalg.inv(R + B.T @ P1 @ B) @ B.T @ P1 @ A

# Compare P and P1, K and K1
norm_P_diff = np.linalg.norm(P - P1, 'fro')
norm_K_diff = np.linalg.norm(K - K1, 'fro')

print(norm_P_diff, norm_K_diff)


# Monte Carlo simulation with Ns noise sequence samples
Ns = 100
T = 100
x0 = 40 * np.random.randn(n, 1)
x = np.zeros((n, T + 1, Ns))
u = np.zeros((m, T, Ns))
J = np.zeros((T + 1, Ns))
J[0, :] = (x0.T @ x0).flatten()
w = np.sqrt(1) * np.random.randn(n, T, Ns)

for i in range(Ns):
    x[:, 0, i] = x0.flatten()
    for j in range(1, T + 1):
        u[:, j - 1, i] = K @ x[:, j - 1, i]
        x[:, j, i] = (A + B @ K) @ x[:, j - 1, i] + w[:, j - 1, i]
        J[j, i] = x[:, j, i].T @ Q @ x[:, j, i] + u[:, j - 1, i].T @ R @ u[:, j - 1, i]

# Plot sample trajectory
plt.figure(figsize=(12, 6))
plt.step(range(T + 1), x[:, :, 0].T, linewidth=2)
plt.xlabel('Time')
plt.ylabel('States')
plt.title('Sample State Trajectory')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.step(range(T), u[:, :, 0].T, linewidth=2)
plt.xlabel('Time')
plt.ylabel('Inputs')
plt.title('Sample Input Trajectory')
plt.grid(True)
plt.show()
