# mypy, pytest

def bfs_flow(n, adj_matrix, cap_matrix) -> tuple[int, list[int]]:
    s: int = 0 # Sources
    t: int = n - 1 # Sink
    queue: list[tuple[int, float]] = [] # [(index, flow)]
    queue.append( (s, float('inf')) )

    parent: list[int] = [-1 for i in range(n)]
    parent[0] = -2

    while (len(queue) != 0):
        current, flow = queue.pop(0)

        # Esto siempre hace el camino más corto al ser BFS
        next_nodes = [ i for i in range(n) if adj_matrix[current][i] == 0]
        for node in next_nodes:
            if parent[node] == -1 and cap_matrix[current][node] != 0:
                parent[node] = current
                new_flow = min(flow, cap_matrix[current][node])
                if node == t:
                    return new_flow, parent
                queue.append( (node, new_flow) )
    return 0, parent


def edmonds_karp(n, adj_matrix, cap_matrix) -> int:
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
Este grafo es el ejemplo del sitio: https://cp-algorithms.com/graph/edmonds_karp.html
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