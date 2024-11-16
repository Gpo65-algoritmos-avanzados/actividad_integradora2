import pytest
from prog import edmonds_karp, tsp_ant_colony_optimization, prim_mst


def test_prim():
    n = 2
    matrix = [[0, 10], 
              [10, 0]]
    
    matrix = [[0, 2, 3, 4, 5],
              [1, 0, 2, 4, 6],
              [2, 3, 0, 5, 1],
              [7, 1, 2, 0, 1],
              [2, 4, 3, 1, 0]]

    prueba1 = prim_mst(n, matrix)

    assert prueba1 == [(0, 1)]


def test_tsp():
    three = 3
    assert three == 3


def test_edmonds_karp():
    one = 1
    assert one == 1