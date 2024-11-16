import pytest
from prog import edmonds_karp, tsp_ant_colony_optimization, prim_mst
import numpy as np

import sys

"""
E2. Actividad Integradora 2 

Ángel Orlando Vázquez Morales A01659000
Sergio Morales González A01657493
Miguel Ángel Ogando Bautista A01663053

"""

def test_prim():
    n1 = 2
    matrix1 = [[0, 10], 
              [10, 0]]

    n2 = 5
    matrix2 = [[0, 2, 3, 4, 5],
               [1, 0, 3, 4, 5],
               [1, 2, 0, 4, 5],
               [1, 2, 3, 0, 5],
               [1, 2, 3, 4, 0]]

    prueba1 = prim_mst(n1, matrix1)
    prueba2 = prim_mst(n2, matrix2)

    

    assert prueba1 == [(0, 1)]
    assert prueba2 == [(0, 1),
                       (0, 2),
                       (0, 3),
                       (0, 4)]


def test_edmonds_karp():
    # Ejemplo 1: https://cp-algorithms.com/graph/edmonds_karp.html
    n1 = 6
    adj_matrix1 = [ [sys.maxsize]*n1 for _ in range(n1) ] 
    cap_matrix1 = [ [0]*n1 for _ in range(n1) ]

    adj_matrix1[0][1] = 1
    adj_matrix1[0][4] = 1
    adj_matrix1[1][0] = 1
    adj_matrix1[4][0] = 1

    adj_matrix1[1][2] = 1
    adj_matrix1[1][3] = 1
    adj_matrix1[2][1] = 1
    adj_matrix1[3][1] = 1

    adj_matrix1[2][5] = 1
    adj_matrix1[5][2] = 1

    adj_matrix1[3][2] = 1
    adj_matrix1[3][5] = 1
    adj_matrix1[2][3] = 1
    adj_matrix1[5][3] = 1

    adj_matrix1[4][1] = 1
    adj_matrix1[4][3] = 1
    adj_matrix1[1][4] = 1
    adj_matrix1[3][4] = 1

    cap_matrix1[0][1] = 7
    cap_matrix1[0][4] = 4
    cap_matrix1[1][2] = 5
    cap_matrix1[1][3] = 3
    cap_matrix1[2][5] = 8
    cap_matrix1[3][5] = 5
    cap_matrix1[4][1] = 3
    cap_matrix1[4][3] = 2

    results1 = edmonds_karp(n1, adj_matrix1, cap_matrix1)
    assert results1 == 10

    # Ejemplo 2: 
    n2 = 6

    adj_matrix2 = [ [sys.maxsize]*n2 for _ in range(n2) ] 
    cap_matrix2 = [ [0]*n2 for _ in range(n2) ]

    adj_matrix2[0][1] = 1
    adj_matrix2[0][2] = 1
    adj_matrix2[1][0] = 1
    adj_matrix2[2][0] = 1

    adj_matrix2[1][3] = 1
    adj_matrix2[1][4] = 1
    adj_matrix2[3][1] = 1
    adj_matrix2[4][1] = 1

    adj_matrix2[2][1] = 1
    adj_matrix2[2][4] = 1
    adj_matrix2[1][2] = 1
    adj_matrix2[4][2] = 1

    adj_matrix2[3][4] = 1
    adj_matrix2[3][5] = 1
    adj_matrix2[4][3] = 1
    adj_matrix2[5][3] = 1

    adj_matrix2[4][5] = 1
    adj_matrix2[5][4] = 1

    cap_matrix2[0][1] = 3
    cap_matrix2[0][2] = 7

    cap_matrix2[1][3] = 3
    cap_matrix2[1][4] = 4

    cap_matrix2[2][1] = 5
    cap_matrix2[2][4] = 3

    cap_matrix2[3][4] = 3
    cap_matrix2[3][5] = 2

    cap_matrix2[4][5] = 6

    results2 = edmonds_karp(n2, adj_matrix2, cap_matrix2)
    assert results2 == 8

    
def test_tsp():
    # Example 1: Simple 4-node graph
    n1 = 4
    distance_matrix1 = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]

    # Expected approximate solution: a valid TSP cycle (e.g., 0 -> 1 -> 3 -> 2 -> 0)
    tsp_route1 = tsp_ant_colony_optimization(np.array(distance_matrix1), n_ants=10, n_iterations=100)

    # Check if the result is a valid cycle
    tsp_nodes1 = set(edge[0] for edge in tsp_route1) | set(edge[1] for edge in tsp_route1)
    assert tsp_nodes1 == {0, 1, 2, 3}

    # Check that the route length is close to the optimal solution (10 + 25 + 30 + 15 = 80)
    route_length1 = sum(distance_matrix1[edge[0]][edge[1]] for edge in tsp_route1)
    assert abs(route_length1 - 80) <= 5  # Allow for small deviations due to heuristic optimization

    # Example 2: Simple 3-node graph
    n2 = 3
    distance_matrix2 = [
        [0, 5, 9],
        [5, 0, 6],
        [9, 6, 0]
    ]

    # Expected approximate solution: a valid TSP cycle (e.g., 0 -> 1 -> 2 -> 0)
    tsp_route2 = tsp_ant_colony_optimization(np.array(distance_matrix2), n_ants=10, n_iterations=50)

    # Check if the result is a valid cycle
    tsp_nodes2 = set(edge[0] for edge in tsp_route2) | set(edge[1] for edge in tsp_route2)
    assert tsp_nodes2 == {0, 1, 2}

    # Check that the route length is close to the optimal solution (5 + 6 + 9 = 20)
    route_length2 = sum(distance_matrix2[edge[0]][edge[1]] for edge in tsp_route2)
    assert abs(route_length2 - 20) <= 2  # Allow for small deviations
