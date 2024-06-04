#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 21:14:31 2024

"""

import numpy as np
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.best_position = position
        self.best_value = float('inf')

def objective_function(x):
    # Définissez ici votre fonction à minimiser
    return x**2  # Exemple de fonction quadratique

def PSO(objective_function, n_particles, n_iterations, bounds):
    particles = []
    global_best_position = None
    global_best_value = float('inf')
    alpha = 0.8
    beta = 0.8
    inertia = 0.8

    # Initialisation des particules
    for _ in range(n_particles):
        position = np.random.uniform(bounds[0], bounds[1])
        velocity = np.random.uniform(-0.1, 0.1)
        particles.append(Particle(position, velocity))

    # Algorithme PSO
    fig, ax = plt.subplots()
    x = np.linspace(bounds[0], bounds[1], 100)
    y = objective_function(x)
    ax.plot(x, y, label='Fonction à minimiser')
    
    for i in range(n_iterations):
        for particle in particles:
            value = objective_function(particle.position)
            if value < particle.best_value:
                particle.best_value = value
                particle.best_position = particle.position

            if value < global_best_value:
                global_best_value = value
                global_best_position = particle.position

        for particle in particles:
            r1, r2 = np.random.rand(), np.random.rand()
            particle.velocity = (inertia * particle.velocity) + (alpha * r1 * (particle.best_position - particle.position)) + (beta * r2 * (global_best_position - particle.position))
            particle.position += particle.velocity

            # Gérer les rebonds sur les bords du domaine
            if particle.position < bounds[0]:
                particle.position = bounds[0]
                particle.velocity *= -1
            elif particle.position > bounds[1]:
                particle.position = bounds[1]
                particle.velocity *= -1

        if i % (n_iterations/10) == 0:  # Plot every 10 iterations
            #x=np.linspace(min(particle.position for particle in particles),max(particle.position for particle in particles) , 100)
            x = np.linspace(bounds[0], bounds[1], 100)
            plt.scatter([particle.position for particle in particles], [objective_function(particle.position) for particle in particles], color='red', alpha=0.5)
            plt.plot(x,objective_function(x))
            plt.title(f'Iteration {i+1}')
            plt.xlabel('Position')
            plt.ylabel('Valeur de la fonction')
            
            plt.show()

    return global_best_position, global_best_value

# Exemple d'utilisation
if __name__ == "__main__":
    bounds = (-10, 10)  # Intervalle de recherche
    n_particles = 30  # Nombre de particules
    n_iterations = 150  # Nombre d'itérations

    minimum_position, minimum_value = PSO(objective_function, n_particles, n_iterations, bounds)
    print("Minimum trouvé à la position:", minimum_position)
    print("Valeur minimale de la fonction:", minimum_value)
