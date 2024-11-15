import sys

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
                if not visited[v] and distance_matrix[u][v] < dis_min[v]:
                    dis_min[v] = distance_matrix[u][v]
                    parent[v] = u
    return mst
