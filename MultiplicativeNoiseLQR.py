import numpy as np
import matplotlib.pyplot as plt

# Utility function for multiplicative noise variance terms
def MultSum(X, z, Zz):
    Z = np.zeros((Zz.shape[1], Zz.shape[1]))
    for i in range(len(z)):
        Z = Z + z[i] * Zz[:, :, i].T @ X @ Zz[:, :, i]

    return Z


# Problem data
n = 4
m = 2

# Nominal system matrices
A = np.random.randn(n, n)
A = 0.8 * A / np.max(np.abs(np.linalg.eigvals(A)))

B = np.random.randn(n, m)

# Multiplicative noise variances for A and B
SigmaA = np.random.randn(n ** 2, n ** 2)
SigmaA = SigmaA @ SigmaA.T
SigmaA = 0.1 * SigmaA / np.max(np.linalg.eigvals(SigmaA))

a, Va = np.linalg.eig(SigmaA)
Aa = np.zeros((n, n, n ** 2))
for i in range(n ** 2):
    Aa[:, :, i] = Va[:, i].reshape(n, n)

SigmaB = np.random.randn(n * m, n * m)
SigmaB = SigmaB @ SigmaB.T
SigmaB = 0.1 * SigmaB / np.max(np.linalg.eigvals(SigmaB))

b, Vb = np.linalg.eig(SigmaB)
Bb = np.zeros((n, m, n * m))
for i in range(n * m):
    Bb[:, :, i] = Vb[:, i].reshape(n, m)

Q = np.eye(n)
QT = np.eye(n)
R = np.eye(m)

T = 30

# Finite horizon dynamic programming algorithm
P = np.zeros((n, n, T + 1))
K = np.zeros((m, n, T))

P[:, :, T] = QT

for i in range(T - 1, -1, -1):
    P[:, :, i] = Q + MultSum(P[:, :, i + 1], a, Aa) + A.T @ (P[:, :, i + 1] - P[:, :, i + 1] @ B @ np.linalg.inv(
                      B.T @ P[:, :, i + 1] @ B + R) @ B.T @ P[:, :, i + 1]) @ A

    K[:, :, i] = -np.linalg.inv(
        R + B.T @ P[:, :, i + 1] @ B + MultSum(P[:, :, i + 1], b, Bb)) @ B.T @ P[:, :, i + 1] @ A

# Infinite horizon
Pinf = np.zeros((n, n))
Pplus = np.random.randn(n, n)

# Value iteration recursion
t = 0
while np.linalg.norm(Pinf - Pplus, 'fro') > 1e-6:
    t += 1
    Pinf = Pplus
    Pplus = Q + A.T @ Pinf @ A + MultSum(Pinf, a, Aa) - A.T @ Pinf @ B @ np.linalg.inv(
        R + B.T @ Pinf @ B + MultSum(Pinf, b, Bb)) @ B.T @ Pinf @ A

Kinf = -np.linalg.inv(
    R + B.T @ Pinf @ B + MultSum(Pinf, b, Bb)) @ B.T @ Pinf @ A

# Plot feedback gains
time_steps = np.arange(T)
for j in range(m):
    plt.plot(time_steps, K[j, 0, :], label=f'(K_t)_{j + 1}1', linewidth=2)

plt.xlabel('Time')
plt.ylabel('Feedback Gains')
plt.legend()
plt.grid(True)
plt.show()