import numpy as np
import matplotlib.pyplot as plt

# Problem data
P = np.array([
    [0.35, 0.65, 0, 0, 0, 0, 0, 0],
    [0.35, 0, 0.65, 0, 0, 0, 0, 0],
    [0, 0.35, 0, 0.65, 0, 0, 0, 0],
    [0, 0, 0.35, 0, 0.65, 0, 0, 0],
    [0, 0, 0, 0.35, 0, 0.65, 0, 0],
    [0, 0, 0, 0, 0.99, 0, 0.01, 0],
    [0, 0, 0, 0, 0, 0.01, 0, 0.99],
    [0, 0, 0, 0, 0, 0, 0.35, 0.65]
])

# Simulate
x0 = 0
T = 1000
x = np.zeros(T + 1, dtype=int)
x[0] = x0

# Define a function to sample from a discrete distribution
def sample(probabilities):
    return np.random.choice(len(probabilities), p=probabilities)

for t in range(T):
    # Draw a sample from the discrete distribution corresponding to the
    # transition matrix row
    x[t + 1] = sample(P[x[t], :])

# Plot the results
plt.plot(x, 'o-')
plt.xlabel('Time Step')
plt.ylabel('State')
plt.title('Metastable Birth-Death Chain')
plt.show()
