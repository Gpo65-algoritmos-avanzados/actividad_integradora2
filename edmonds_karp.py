from typing import List, Tuple

def bfs_flow(n: int, adj_matrix: List[List[float]], cap_matrix: List[List[int]]) -> Tuple[int, List[int]]:
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
    queue: list[tuple[int, float]] = []
    queue.append( (s, float('inf')) )

    parent: list[int] = [-1 for _ in range(n)]
    parent[0] = -2

    while (len(queue) != 0):
        current, flow = queue.pop(0)

        next_nodes = [ i for i in range(n) if adj_matrix[current][i] == 0]
        for node in next_nodes:
            if parent[node] == -1 and cap_matrix[current][node] != 0:
                parent[node] = current
                new_flow = min(int(flow), cap_matrix[current][node])
                if node == t:
                    return new_flow, parent
                queue.append( (node, new_flow) )
    return 0, parent


def edmonds_karp(n: int, adj_matrix: List[List[float]], cap_matrix: List[List[int]]) -> int:
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

"""
Este grafo es el ejemplo del sitio: https://cp-algorithms.com/graph/edmonds_karp.html
"""
def grafo1():
    adj_matrix = [ [float('inf')]*n for _ in range(n) ] 
    cap_matrix = [ [float('inf')]*n for _ in range(n) ]

    adj_matrix[0][1] = 0
    adj_matrix[0][4] = 0
    adj_matrix[1][0] = 0
    adj_matrix[4][0] = 0

    adj_matrix[1][2] = 0
    adj_matrix[1][3] = 0
    adj_matrix[2][1] = 0
    adj_matrix[3][1] = 0

    adj_matrix[2][5] = 0
    adj_matrix[5][2] = 0

    adj_matrix[3][2] = 0
    adj_matrix[3][5] = 0
    adj_matrix[2][3] = 0
    adj_matrix[5][3] = 0

    adj_matrix[4][1] = 0
    adj_matrix[4][3] = 0
    adj_matrix[1][4] = 0
    adj_matrix[3][4] = 0

    cap_matrix[0][1] = 7
    cap_matrix[0][4] = 4
    cap_matrix[1][2] = 5
    cap_matrix[1][3] = 3
    cap_matrix[2][5] = 8
    cap_matrix[3][5] = 5
    cap_matrix[4][1] = 3
    cap_matrix[4][3] = 2

    return adj_matrix, cap_matrix

"""
Este grafo es el ejemplo del sitio: https://www.w3schools.com/dsa/dsa_algo_graphs_edmondskarp.php
"""
def grafo2():
    adj_matrix = [ [float('inf')]*n for _ in range(n) ] 
    cap_matrix = [ [float('inf')]*n for _ in range(n) ]

    adj_matrix[0][1] = 0
    adj_matrix[0][2] = 0
    adj_matrix[1][0] = 0
    adj_matrix[2][0] = 0

    adj_matrix[1][3] = 0
    adj_matrix[1][4] = 0
    adj_matrix[3][1] = 0
    adj_matrix[4][1] = 0

    adj_matrix[2][1] = 0
    adj_matrix[2][4] = 0
    adj_matrix[1][2] = 0
    adj_matrix[4][2] = 0

    adj_matrix[3][4] = 0
    adj_matrix[3][5] = 0
    adj_matrix[4][3] = 0
    adj_matrix[5][3] = 0

    adj_matrix[4][5] = 0
    adj_matrix[5][4] = 0

    cap_matrix[0][1] = 3
    cap_matrix[0][2] = 7

    cap_matrix[1][3] = 3
    cap_matrix[1][4] = 4

    cap_matrix[2][1] = 5
    cap_matrix[2][4] = 3

    cap_matrix[3][4] = 3
    cap_matrix[3][5] = 2

    cap_matrix[4][5] = 6

    return adj_matrix, cap_matrix

if __name__ == "__main__":
    n = 6
    adj1, cap1 = grafo1()
    max_flow = edmonds_karp(6, adj1, cap1)
    print(max_flow)

    adj2, cap2 = grafo2()
    max_flow = edmonds_karp(6, adj2, cap2)
    print(max_flow)