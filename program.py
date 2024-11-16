from typing import List, Tuple
import numpy as np
import sys

"""
E2. Actividad Integradora 2 

Ángel Orlando Vázquez Morales A01659000
Sergio Morales González A01657493
Miguel Ángel Ogando Bautista A01663053

"""

def prim_mst(n: int, distance_matrix: list[list[int]]) -> list[tuple[int, int]]:
    """
    Computes the Minimum Spanning Tree (MST) of a graph using Prim's algorithm.

    Parameters:
    n (int): The number of vertices in the graph.
    distance_matrix (list[list[int]]): A 2D list representing the adjacency matrix of the graph,
                                       where distance_matrix[i][j] is the weight of the edge
                                       between vertices i and j.

    Returns:
    list[tuple[int, int]]: A list of tuples representing the edges in the MST. Each tuple
                           contains two integers, representing the vertices connected by the edge.
    """
    mst = []
    visited = [False] * n
    dis_min = [sys.maxsize] * n
    parent = [-1] * n
    dis_min[0] = 0
    for _ in range(n):
        min_dis = sys.maxsize
        u = -1
        for i in range(n):
            if not visited[i] and dis_min[i] < min_dis:
                min_dis = dis_min[i]
                u = i
        visited[u] = True
        if parent[u] != -1:
            mst.append((parent[u], u))
        for v in range(n):
            if not visited[v] and 0 < distance_matrix[u][v] < dis_min[v]:
                dis_min[v] = distance_matrix[u][v]
                parent[v] = u
    return mst



def tsp_ant_colony_optimization(
    distance_matrix: np.ndarray, 
    n_ants: int = 10, 
    n_iterations: int = 100, 
    decay: float = 0.5, 
    alpha: float = 1, 
    beta: float = 2
) -> List[Tuple[int, int]]:
    """
    Solves the Traveling Salesman Problem using Ant Colony Optimization.
    Returns an empty array in the case no possible shortest route is found.

    Parameters:
      distance_matrix (np.ndarray): A 2D numpy array representing the distances between each pair of nodes.
      n_ants (int): The number of ants to simulate in each iteration.
      n_iterations (int): The number of iterations to perform.
      decay (float): The rate at which pheromone decays.
      alpha (float): The influence of pheromone on direction choice.
      beta (float): The influence of distance on direction choice.

    Returns:
      path (List[Tuple[int, int]]): A list of tuples representing the shortest route found, with nodes labeled as letters.
    """
    n = distance_matrix.shape[0]
    pheromone = np.ones((n, n))
    shortest_route: list[int] = []
    shortest_route_length = float("inf")

    def compute_route_length(route: List[int]) -> float:
        """Computes the total length of a given route."""
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route) - 1)) + distance_matrix[route[-1], route[0]]

    for _ in range(n_iterations):
        all_routes: list[list[int]] = []
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

    if len(shortest_route) == 0:
        return [] 
    return [(shortest_route[i], shortest_route[(i + 1) % n]) for i in range(n)]


def bfs_flow(n: int, adj_matrix: List[List[int]], cap_matrix: List[List[int]]) -> Tuple[int, List[int]]:
    """
    Implements the bfs algorithm to use for the Edmonds-Karp algorithm, it traverses the graph from the node 0 until the
    last node of the graph is reached, each iteration storing the max amount flow that each node can flow until the last node is found.
    If the last node can't be reached then the algorithm returns 0 as the found flow.

    Parameters:
    - n (int): The number of nodes in the graph.
    - adj_matrix (List[List[float]]): A 2D list representing the adjacency matrix of the graph,
                                       where adj_matrix[i][j] is the weight of the edge
                                       between vertices i and j.
    - cap_matrix (List[List[int]]): A 2D list representing the capacity matrix of the graph,
                                       where cap_matrix[i][j] is the flow capacity of the edge
                                       between vertices i and j.

    Returns:
    - Tuple[int, List[int]]: Returns an integer that represent the flow .
    """
    s: int = 0 # Source index
    t: int = n - 1 # Sink index
    queue: List[Tuple[int, int]] = []
    queue.append( (s, sys.maxsize) )

    parent: List[int] = [-1 for _ in range(n)]
    parent[0] = -2

    while (len(queue) != 0):
        current, flow = queue.pop(0)
        next_nodes = [ i for i in range(n) if adj_matrix[current][i] != 0 and current != i]
        for node in next_nodes:
            if parent[node] == -1 and cap_matrix[current][node] != 0:
                parent[node] = current
                new_flow = min(flow, cap_matrix[current][node])
                if node == t:
                    return new_flow, parent
                queue.append( (node, new_flow) )
    return 0, parent


def edmonds_karp(n: int, adj_matrix: List[List[int]], cap_matrix: List[List[int]]) -> int:
    """
    Implements the Edmonds-Karp algorithm to find the maximum flow in a graph. 
    The algorithm assumes that the source is located at the index 0 and the sink is located at the index n-1.

    Parameters:
      n (int): The number of nodes in the graph.
      adj_matrix (List[List[float]]): A 2D list representing the adjacency matrix of the graph,
                                       where adj_matrix[i][j] is the weight of the edge
                                       between vertices i and j.
      cap_matrix (List[List[int]]): A 2D list representing the capacity matrix of the graph,
                                       where cap_matrix[i][j] is the flow capacity of the edge
                                       between vertices i and j.

    Returns:
      flow (int): An integer that represents the maximum flow possible between the first node (index 0) and the last node (index n - 1) of the given graph.
    """
    max_flow: int = 0
    s: int = 0
    t: int = n - 1

    flow: int = -1
    while True:
        flow, parent = bfs_flow(n, adj_matrix, cap_matrix)
        if flow == 0: break

        max_flow += flow
        node: int = t
        while node != s:
            prev = parent[node]
            cap_matrix[prev][node] -= flow
            cap_matrix[node][prev] += flow
            node = prev
    return max_flow


def readFile(filename: str = "input.txt") -> Tuple[int, List[List[int]], List[List[int]]]:
    """
    It reads the matrix in the filename specified and returns the amount of nodes, adjacency matrix and capacity matrix.

    Parameters:
      filename (str): Represents the filename to read the matrix.
    
    Returns:
      matrix (Tuple[int, List[List[int]], List[List[int]]]): An integer that represents the maximum flow possible between the first node (index 0) and the last node (index n - 1) of the given graph.
    """
    n: int = 0
    adj_matrix: List[List[int]] = []
    cap_matrix: List[List[int]] = []
    with open(filename, 'r') as file:
        n_str: str = file.readline().strip()
        n = int(n_str)

        # Reads an empty line that separates the adjacency matrix. 
        file.readline()

        for _ in range(n):
            row: str = file.readline().strip()
            row_strs: List[str] = row.split(" ")
            row_filtered: List[str] = list(filter(lambda x: x != "", row_strs))

            row_numbers: List[int] = [int(n) for n in row_filtered]
            adj_matrix.append(row_numbers)

        # Reads an empty line that separates the capacity matrix. 
        file.readline()

        for _ in range(n):
            row = file.readline().strip()
            row_strs = row.split(" ")
            row_filtered = list(filter(lambda x: x != "", row_strs))

            row_numbers = [int(n) for n in row_filtered]
            cap_matrix.append(row_numbers)
    return n, adj_matrix, cap_matrix


if __name__ == "__main__":
    n, adj_matrix, cap_matrix = readFile(filename="input.txt")
    adj_matrix_np = np.asmatrix(adj_matrix)

    #prim = prim_mst(n, adj_matrix)
    #output = ", ".join(prim)
    #print(output)
    prim_result = prim_mst(n, adj_matrix)
    prim_strings: List[str] = list(map(lambda x: str(x), prim_result))

    print(", ".join([ str(pair) for pair in prim_strings ]))
    print()

    tsp_result = tsp_ant_colony_optimization(adj_matrix_np)
    tsp_strings: List[str] = list(map(lambda x: str(x), tsp_result))

    print(", ".join([ str(pair) for pair in tsp_strings ]))
    print()

    print(edmonds_karp(n, adj_matrix, cap_matrix))