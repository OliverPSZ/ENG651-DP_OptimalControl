import numpy as np
import matplotlib.pyplot as plt

# Define the transition matrix
# Replace this with your specific transition matrix
# Example: A simple 2-state chain
transition_matrix = np.array([[0.8, 0.2],
                               [0.3, 0.7]])

# Number of time steps
T = 10

# Initial state
initial_state = 0

# Simulate the Markov chain
# np.random.seed(0)  # Seed for reproducibility
current_state = initial_state
sample_path = [current_state]

for t in range(T):
    # Generate a random number to determine the next state
    next_state = np.random.choice(len(transition_matrix), p=transition_matrix[current_state])
    sample_path.append(next_state)
    current_state = next_state

# Print the sample path
print("Sample Path:", sample_path)

# Plot the sample path
plt.plot(range(T + 1), sample_path, marker='o', linestyle='-')
plt.xlabel('Time Step')
plt.ylabel('State')
plt.title('Sample Path of the Markov Chain')
plt.grid(True)
plt.show()
