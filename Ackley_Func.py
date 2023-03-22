"""
f(x) = -20 * exp(-0.2 * sqrt((1/d) * sum(xi^2))) - exp((1/d) * sum(cos(2 * pi * xi))) + 20 + exp(1)
"""
import numpy as np

def ackley(xi, d):
   return (-20 * np.exp(-0.2 * np.sqrt((1/d) * sum(xi**2))) - np.exp((1/d) * sum(np.cos(2 * np.pi * xi))) + 20 + np.exp(1))

class Particle():
    def __init__(self, dim, xmin, xmax):
        self.pos = np.random.uniform(low=xmin, high=xmax, size=dim)
        self.vel = np.zeros(dim)
        self.best_pos = self.pos.copy()
        self.best_fit = ackley(self.pos, dim)
    def update(self, xmin, xmax, global_best_position, omega, phi_p, phi_g):
        r_p, r_g = np.random.rand(2, len(self.pos))
        self.vel = omega*self.vel + phi_p*r_p*(self.best_pos - self.pos) + phi_g*r_g*(global_best_position - self.pos)
        self.pos = np.clip(self.pos + self.vel, xmin, xmax)
        fit = ackley(self.pos, dim)
        if fit < self.best_fit:
            self.best_pos = self.pos.copy()
            self.best_fit = fit

def pso(dim, n_particles, n_iterations, xmin, xmax, omega, phi_p, phi_g):
    particles = [Particle(dim, xmin, xmax) for _ in range(n_particles)]
    global_best_position = particles[0].pos.copy()
    global_best_fitness = ackley(global_best_position, dim)
    for i in range(1, n_particles):
        if particles[i].best_fit < global_best_fitness:
            global_best_position = particles[i].best_pos.copy()
            global_best_fitness = particles[i].best_fit
    for _ in range(n_iterations):
        for particle in particles:
            particle.update(global_best_position, omega, phi_p, phi_g, xmin, xmax)
            if particle.best_fit < global_best_fitness:
                global_best_position = particle.best_pos.copy()
                global_best_fitness = particle.best_fit
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