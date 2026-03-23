import numpy as np

# Example 1: Ergodic Chain
P1 = np.array([
    [0.5, 0.5, 0, 0],
    [0.3, 0.4, 0.3, 0],
    [0, 0.3, 0.4, 0.3],
    [0, 0, 0.5, 0.5]
])

d0_1 = np.random.rand(4)
d0_1 /= np.sum(d0_1)

T = 100
d_1 = np.zeros((T, 4))
d_1[0, :] = d0_1

# Distribution propagation
for t in range(T - 1):
    d_1[t + 1, :] = d_1[t, :] @ P1

dT_1 = d_1[-1, :]


# Example 2: Absorbing Chain
P2 = np.array([
    [0.3, 0.3, 0, 0, 0, 0.4, 0],
    [0.4, 0.3, 0.3, 0, 0, 0, 0],
    [0, 0.4, 0.3, 0.3, 0, 0, 0],
    [0, 0, 0.4, 0.3, 0.3, 0, 0],
    [0, 0, 0, 0.4, 0.3, 0, 0.3],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1]
])

d0_2 = np.random.rand(7)
d0_2 /= np.sum(d0_2)

T = 100
d_2 = np.zeros((T, 7))
d_2[0, :] = d0_2

# Distribution propagation
for t in range(T - 1):
    d_2[t + 1, :] = d_2[t, :] @ P2

dT_2 = d_2[-1, :]

# Example 3: Periodic Chain
P3 = np.array([
    [0, 1],
    [1, 0]
])

d0_3 = np.array([1, 0])

T = 100
d_3 = np.zeros((T, 2))
d_3[0, :] = d0_3

# Distribution propagation
for t in range(T - 1):
    d_3[t + 1, :] = d_3[t, :] @ P3

dT_3 = d_3[-1, :]

print("Example 1 Ergodic Chain Result:")
print("Initial Distribution:", d0_1)
print("Final Distribution:", dT_1)

print("\nExample 2 Absorbing Chain Result:")
print("Initial Distribution:", d0_2)
print("Final Distribution:", dT_2)

print("\nExample 3 Periodic Chain Result:")
print("Initial Distribution:", d0_3)
print("Final Distribution:", dT_3)
