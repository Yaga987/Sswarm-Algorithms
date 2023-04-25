import random

# Define the function to be minimized
def f(x):
    return (x+13)**2 + (x-9)**2

# Define the bounds of the search space
x_min = -100   # lower bound of x
x_max = 100  # upper bound of x

# Define the parameters of the PSO algorithm
n_particles = 20    # number of particles in the swarm
max_iterations = 50 # maximum number of iterations
w = 0.5            # inertia weight
c1 = 1             # cognitive constant
c2 = 2             # social constant

# Initialize the swarm
particles_x = [random.uniform(x_min, x_max) for _ in range(n_particles)]
particles_v = [0 for _ in range(n_particles)]
particles_p = particles_x.copy()
particles_fp = [f(x) for x in particles_p]
global_best_p = particles_p[particles_fp.index(min(particles_fp))]
global_best_fp = min(particles_fp)

# Run the PSO algorithm
for iteration in range(max_iterations):
    for i in range(n_particles):
        # Update velocity
        particles_v[i] = w*particles_v[i] + c1*random.random()*(particles_p[i]-particles_x[i]) \
                                         + c2*random.random()*(global_best_p-particles_x[i])
        # Update position
        particles_x[i] = particles_x[i] + particles_v[i]
        
        # Check if position is within bounds
        if particles_x[i] < x_min:
            particles_x[i] = x_min
        elif particles_x[i] > x_max:
            particles_x[i] = x_max
            
        # Evaluate fitness
        particles_fp[i] = f(particles_x[i])
        
        # Update personal best
        if particles_fp[i] < f(particles_p[i]):
            particles_p[i] = particles_x[i]
            
        # Update global best
        if particles_fp[i] < global_best_fp:
            global_best_p = particles_x[i]
            global_best_fp = particles_fp[i]
            
    # Print progress
    print("Iteration %d: Best fitness = %.4f" % (iteration+1, global_best_fp))

# Print the result
print("Minimum value found at x = %.4f" % global_best_p)
print("Minimum function value = %.4f" % global_best_fp)
