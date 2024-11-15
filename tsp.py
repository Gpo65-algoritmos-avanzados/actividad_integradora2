import numpy as np
from typing import List, Tuple

def tsp_ant_colony_optimization(
    distance_matrix: np.ndarray, 
    n_ants: int = 10, 
    n_iterations: int = 100, 
    decay: float = 0.5, 
    alpha: float = 1, 
    beta: float = 2
) -> List[Tuple[str, str]]:
    """
    Solves the Traveling Salesman Problem using Ant Colony Optimization.

    Parameters:
    - distance_matrix (np.ndarray): A 2D numpy array representing the distances between each pair of nodes.
    - n_ants (int): The number of ants to simulate in each iteration.
    - n_iterations (int): The number of iterations to perform.
    - decay (float): The rate at which pheromone decays.
    - alpha (float): The influence of pheromone on direction choice.
    - beta (float): The influence of distance on direction choice.

    Returns:
    - List[Tuple[str, str]]: A list of tuples representing the shortest route found, with nodes labeled as letters.
    """
    n = distance_matrix.shape[0]
    pheromone = np.ones((n, n))
    shortest_route = None
    shortest_route_length = float("inf")

    def compute_route_length(route: List[int]) -> float:
        """Computes the total length of a given route."""
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route) - 1)) + distance_matrix[route[-1], route[0]]

    for iteration in range(n_iterations):
        all_routes = []
        for _ in range(n_ants):
            route = [0]
            visited = set(route)
            while len(route) < n:
                current = route[-1]
                probabilities = []
                for j in range(n):
                    if j not in visited:
                        prob = (pheromone[current, j] ** alpha) * ((1 / distance_matrix[current, j]) ** beta)
                        probabilities.append((j, prob))
                next_node = max(probabilities, key=lambda x: x[1])[0]
                route.append(next_node)
                visited.add(next_node)
            all_routes.append(route)
        
        for route in all_routes:
            route_length = compute_route_length(route)
            if route_length < shortest_route_length:
                shortest_route = route
                shortest_route_length = route_length

        for i in range(n):
            for j in range(i + 1, n):
                pheromone[i, j] *= (1 - decay)

        for route in all_routes:
            route_length = compute_route_length(route)
            for i in range(n):
                j = (i + 1) % n
                pheromone[route[i], route[j]] += 1 / route_length

    return [(chr(65 + shortest_route[i]), chr(65 + shortest_route[(i + 1) % n])) for i in range(n)]
