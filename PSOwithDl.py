import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Load and preprocess the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype('float32')
y = mnist.target.astype('int64')
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the fitness function
def fitness_function(position):
    n_hidden = int(position[0])
    hidden_size = int(position[1])
    lr = 10 ** position[2]
    clf = MLPClassifier(hidden_layer_sizes=(n_hidden, hidden_size), 
                        learning_rate_init=lr, max_iter=50)
    clf.fit(X_train, y_train)
    score = clf.score(X_val, y_val)
    return -score

# Define the PSO algorithm
def pso(fitness_function, n_particles, dimensions, bounds, iters, c1, c2, w):
    # Initialize the particles and velocities
    particles = np.random.uniform(bounds[0], bounds[1], (n_particles, dimensions))
    velocities = np.zeros((n_particles, dimensions))
    best_positions = particles.copy()
    best_scores = np.zeros(n_particles)
    global_best_position = particles[0].copy()
    global_best_score = float('inf')
    history = np.zeros(iters)
    
    # Iterate over the specified number of generations
    for i in range(iters):

        print(f'Iteration : {i + 1}')

        # Evaluate the fitness of each particle
        scores = np.array([fitness_function(p) for p in particles])
        
        # Update the best positions and scores
        for j in range(n_particles):
            if scores[j] < best_scores[j]:
                best_positions[j] = particles[j].copy()
                best_scores[j] = scores[j]
            if scores[j] < global_best_score:
                global_best_position = particles[j].copy()
                global_best_score = scores[j]
        
        # Update the velocities and positions of the particles
        r1 = np.random.rand(n_particles, dimensions)
        r2 = np.random.rand(n_particles, dimensions)
        velocities = w * velocities + c1 * r1 * (best_positions - particles) + c2 * r2 * (global_best_position - particles)
        particles += velocities
        
        # Ensure the particles stay within the specified bounds
        particles = np.maximum(particles, bounds[0])
        particles = np.minimum(particles, bounds[1])
        
        # Record the best score of this generation
        history[i] = global_best_score
    
    return global_best_position, global_best_score, history

# Run the PSO algorithm
n_particles = 20
dimensions = 3
bounds = (np.array([1, 1, -5]), np.array([100, 500, 0]))
iters = 5
c1 = 0.5
c2 = 0.3
w = 0.9
best_position, best_score, history = pso(fitness_function, n_particles, dimensions, bounds, iters, c1, c2, w)

# Print the best hyperparameters and accuracy
n_hidden, hidden_size, lr = best_position
n_hidden = int(n_hidden)
hidden_size = int(hidden_size)
lr = 10 ** lr
print(f"Best hidden layer configuration: ({n_hidden}, {hidden_size})")
print(f"Best learning rate: {lr}")
print(f"Best validation accuracy: {1-best_score:.4f}")

# Plot the history of the best scores with mean, max, and min lines
mean_line = np.mean(history)
max_point = (np.argmax(history), np.max(history))
min_point = (np.argmin(history), np.min(history))
plt.plot(history, label='Best Score')
plt.axhline(mean_line, color='red', linestyle='--', label='Mean')
plt.plot(max_point[0], max_point[1], marker='o', color='green', label='Max')
plt.plot(min_point[0], min_point[1], marker='o', color='orange', label='Min')
plt.xlabel('Generation')
plt.ylabel('Best Score')
plt.title('PSO Performance')
plt.legend()
plt.show()