import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    """
    Fonction à minimiser.
    """
    return x**2 + y**2 

def plot_surface_and_particles(f, particles, best_global_position, iteration):
    """
    Affiche la surface de la fonction et les particules.
    """
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
    ax.scatter(particles[:, 0], particles[:, 1], f(particles[:, 0], particles[:, 1]), color='red', label='Particles')
    ax.scatter(best_global_position[0], best_global_position[1], f(best_global_position[0], best_global_position[1]), color='green', label='Global Best')
    ax.set_title('Iteration {}'.format(iteration))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X, Y)')
    ax.legend()
    plt.show()

def particle_swarm_optimization(f, num_particles=50, num_iterations=100, inertia=0.8, alpha=0.5, beta=0.5, xmin=-10, xmax=10, ymin=-10, ymax=10, vmax=0.1, tol=0.0001, plot_interval=10):
    """
    Algorithme des essaims particulaires pour minimiser une fonction de deux variables.
    """
    # Initialisation
    particles = np.random.uniform(low=[xmin, ymin], high=[xmax, ymax], size=(num_particles, 2))
    velocities = np.zeros_like(particles)
    best_positions = particles.copy()
    best_global_position = particles[np.argmin([f(p[0], p[1]) for p in particles])]
    distances = []

    # Boucle d'optimisation
    for iteration in range(num_iterations):
        # Calcul des distances moyennes entre deux itérations successives
        if len(distances) >= 2:
            if np.abs(distances[-1] - distances[-2]) < tol:
                break

        # Mise à jour des distances
        distances.append(np.mean(np.linalg.norm(particles - best_global_position, axis=1)))

        # Mise à jour des vitesses et positions
        for i in range(num_particles):
            r1, r2 = np.random.uniform(0, 1, 2)
            velocities[i] = inertia * velocities[i] + alpha * r1 * (best_positions[i] - particles[i]) + beta * r2 * (best_global_position - particles[i])

            # Vérification des bornes
            velocities[i] = np.clip(velocities[i], -vmax, vmax)
            particles[i] += velocities[i]

            # Vérification des limites
            particles[i][0] = np.clip(particles[i][0], xmin, xmax)
            particles[i][1] = np.clip(particles[i][1], ymin, ymax)

            # Mise à jour des meilleures positions
            if f(particles[i][0], particles[i][1]) < f(best_positions[i][0], best_positions[i][1]):
                best_positions[i] = particles[i].copy()

                if f(particles[i][0], particles[i][1]) < f(best_global_position[0], best_global_position[1]):
                    best_global_position = particles[i].copy()

        # Affichage régulier
        if iteration % plot_interval == 0:
            plot_surface_and_particles(f, particles, best_global_position, iteration)

    # Retour des résultats
    return best_global_position, f(best_global_position[0], best_global_position[1]), distances

# Exemple d'utilisation
best_position, minimum, distances = particle_swarm_optimization(f)
print("Minimum de la fonction:", minimum)
print("Position du minimum:", best_position)
