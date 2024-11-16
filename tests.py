import pytest
from prog import edmonds_karp, tsp_ant_colony_optimization, prim_mst
import sys

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


def test_tsp():
    three = 3
    assert three == 3


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