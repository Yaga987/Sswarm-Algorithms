import numpy as np
import matplotlib.pyplot as plt

# Func
def Sphere(x):
    return np.sum(np.square(x))

# Define Parameters
d = 10 # Dimension
xMin, xMax = -100, 100 # Border
vMin, vMax = -0.2*(xMax - xMin), 0.2*(xMax - xMin)  # Velocity
MaxIt = 3000 # Max Iteration
ps = 10 # Population size
c1, c2 = 2, 2
w = 0.9 - ((0.9 - 0.4)/MaxIt)*np.linspace(0, MaxIt, MaxIt)

def limitV(V):
    for i in range(len(V)):
        if V[i] > vMax:
            V[i] = vMax
        if V[i] > vMin:
            V[i] = vMin
    return V

def limitX(X):
    for i in range(len(X)):
        if X[i] > xMax:
            X[i] = xMax
        if X[i] > xMin:
            X[i] = xMin
    return X

def Optimazation():
    class Particle():
        def __init__(self):
            self.position = np.random.uniform(xMin, 50, [ps, d])
            self.velocity = np.random.uniform(vMin, vMax, [ps, d])
            self.cost = np.zeros(ps)
            self.cost[:] = Sphere(self.position[:])
            self.pbest = np.copy(self.position)  # Best pos
            self.pbest_cost = np.copy(self.cost)
            self.index = np.argmin(self.pbest_cost)
            self.gbest = self.pbest[self.index]  # Global Best
            self.gbest_cost = self.pbest_cost[self.index]
            self.BestCost = np.zeros(MaxIt)
        def Evaluate(self):
            for it in range(MaxIt):
                for i in range(ps):
                    self.velocity[i] = (w[it]*self.velocity[i]
                                        + c1*np.random.rand(d)*(self.pbest[i] - self.position[i])
                                        + c2*np.random.rand(d)*(self.gbest[i] - self.position[i]))
                    self.velocity[i] = limitV(self.velocity[i])
                    self.position[i] = self.position[i] + self.velocity[i]
                    self.position[i] = limitX(self.position[i])
                    self.cost[i] = Sphere(self.position[i])
                    if self.cost[i] < self.pbest_cost[i]:
                        self.pbest[i] = self.position[i]
                        self.pbest_cost[i] = self.cost[i]
                        if self.pbest_cost[i] < self.gbest_cost:
                            self.gbest = self.pbest[i]
                            self.gbest_cost = self.pbest_cost[i]
                self.BestCost[it] = self.gbest_cost
        def Plot(self):
            plt.semilogy(self.BestCost)
            plt.show()
            print(f'Best fitness value = {self.gbest_cost}')
    a = Particle()
    a.Evaluate()
    a.Plot()

Optimazation()