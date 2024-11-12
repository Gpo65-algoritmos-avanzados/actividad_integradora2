import sys

def prim_mst(n, distance_matrix):
    mst = []
    visited = [False] * n
    dis_min=[sys.maxsize]*n
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
