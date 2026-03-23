import numpy as np
import matplotlib.pyplot as plt

# Problem data
n = 5
m = 2
T = 30

x0 = 40 * np.random.randn(n, 1)

A = np.random.randn(n, n)
A = A / (0.01 + max(abs(np.linalg.eig(A)[0])))
B = np.random.randn(n, m)

Q = np.eye(n)
QT = np.eye(n)
R = 100 * np.eye(m)

W = 0.1 * np.eye(n)

# Optimal closed-loop feedback policy
P = np.zeros((n, n, T + 1))
r = np.zeros(T + 1)
K = np.zeros((m, n, T))

P[:, :, T] = QT

for i in range(T - 1, -1, -1):
    P[:, :, i] = Q + A.T @ (P[:, :, i + 1] - P[:, :, i + 1] @ B @ np.linalg.inv(B.T @ P[:, :, i + 1] @ B + R) @ B.T @ P[:, :, i + 1]) @ A
    r[i] = r[i + 1] + np.trace(P[:, :, i + 1] @ W)
    K[:, :, i] = -np.linalg.inv(B.T @ P[:, :, i + 1] @ B + R) @ B.T @ P[:, :, i + 1] @ A

# Plotting
time_steps = np.arange(T)
for i in range(n):
    for j in range(m):
        plt.plot(time_steps, K[j, i, :], label=f'(K_t)_{j + 1}1', linewidth=2)

plt.xlabel('Time')
plt.ylabel('Feedback Gains')
plt.legend()
plt.grid(True)
plt.show()
