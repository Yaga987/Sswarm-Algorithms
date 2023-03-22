"""
f(x) = An + sum(x^2 - Acos(2pix)), where x is a vector of n real values,
 A is a constant (typically set to 10 or 20), and the minimum value of f(x) is 0, which occurs at x = 0.
"""
import numpy as np

def rastrigin(x, A=10):
    n = len(x)
    return A*n + np.sum(x**2 - A*np.cos(2*np.pi*x))

class Particle:
    def __init__(self, dim, x_min, x_max):
        self.position = np.random.uniform(low=x_min, high=x_max, size=dim)
        self.velocity = np.zeros(dim)
        self.best_position = self.position.copy()
        self.best_fitness = rastrigin(self.position)
    
    def update_velocity(self, global_best_position, omega, phi_p, phi_g):
        r_p, r_g = np.random.rand(2, len(self.position))
        self.velocity = omega*self.velocity + phi_p*r_p*(self.best_position - self.position) + phi_g*r_g*(global_best_position - self.position)
    
    def update_position(self, x_min, x_max):
        self.position = np.clip(self.position + self.velocity, x_min, x_max)
        fitness = rastrigin(self.position)
        if fitness < self.best_fitness:
            self.best_position = self.position.copy()
            self.best_fitness = fitness

def pso(dim, n_particles, n_iterations, x_min, x_max, omega, phi_p, phi_g):
    particles = [Particle(dim, x_min, x_max) for _ in range(n_particles)]
    global_best_position = particles[0].position.copy()
    global_best_fitness = rastrigin(global_best_position)
    for i in range(1, n_particles):
        if particles[i].best_fitness < global_best_fitness:
            global_best_position = particles[i].best_position.copy()
            global_best_fitness = particles[i].best_fitness
    for _ in range(n_iterations):
        for particle in particles:
            particle.update_velocity(global_best_position, omega, phi_p, phi_g)
            particle.update_position(x_min, x_max)
            if particle.best_fitness < global_best_fitness:
                global_best_position = particle.best_position.copy()
                global_best_fitness = particle.best_fitness
    return global_best_position, global_best_fitness

if __name__ == '__main__':
    dim = 10
    n_particles = 20
    n_iterations = 100
    x_min = -5.12
    x_max = 5.12
    omega = 0.5
    phi_p = 0.5
    phi_g = 0.5
    result = pso(dim, n_particles, n_iterations, x_min, x_max, omega, phi_p, phi_g)
    print('Minimum found at:', result[0])
    print('Minimum value:', result[1])